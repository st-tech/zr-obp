# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Evaluation Class to Streamline OPE."""
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

from .estimators_continuous import BaseContinuousOffPolicyEstimator
from ..types import BanditFeedback
from ..utils import check_confidence_interval_arguments

logger = getLogger(__name__)


@dataclass
class ContinuousOffPolicyEvaluation:
    """Class to conduct OPE using multiple estimators simultaneously.

    Parameters
    -----------
    bandit_feedback: BanditFeedback
        Logged bandit feedback data with continuous actions used to conduct OPE.

    ope_estimators: List[BaseOffPolicyEstimator]
        List of OPE estimators used to evaluate the policy value of evaluation policy.
        Estimators must follow the interface of `obp.ope.BaseContinuousOffPolicyEstimator`.

    Examples
    ----------

    .. code-block:: python

        # a case of implementing OPE (with continuous actions) of an synthetic evaluation policy
        >>> from obp.dataset import (
                SyntheticContinuousBanditDataset,
                linear_reward_funcion_continuous,
                linear_behavior_policy_continuous,
                linear_synthetic_policy_continuous
            )
        >>> from obp.ope import (
                ContinuousOffPolicyEvaluation,
                KernelizedInverseProbabilityWeighting as KernelizedIPW
            )

        # (1) Synthetic Data Generation
        >>> dataset = SyntheticContinuousBanditDataset(
                dim_context=5,
                reward_function=linear_reward_funcion_continuous,
                behavior_policy_function=linear_behavior_policy_continuous,
                random_state=12345,
            )
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(
                n_rounds=10000, min_action_value=-10, max_action_value=10,
            )

        # (2) Synthetic Evaluation Policy
        >>> action_by_evaluation_policy = linear_synthetic_policy_continuous(
                context=bandit_feedback["context"]
            )

        # (3) Off-Policy Evaluation
        >>> ope = ContinuousOffPolicyEvaluation(
                bandit_feedback=bandit_feedback,
                ope_estimators=[KernelizedIPW(kernel="epanechnikov", bandwidth=0.02)]
            )
        >>> estimated_policy_value = ope.estimate_policy_values(
                action_by_evaluation_policy=action_by_evaluation_policy,
            )
        >>> estimated_policy_value
        {'kernelized_ipw': 2.2858905015106723}

        # (4) Ground-truth Policy Value of the Synthetic Evaluation Policy
        >>> dataset.calc_ground_truth_policy_value(
                context=bandit_feedback["context"], action=action_by_evaluation_policy
            )
        2.2893029243895215

    """

    bandit_feedback: BanditFeedback
    ope_estimators: List[BaseContinuousOffPolicyEstimator]

    def __post_init__(self) -> None:
        """Initialize class."""
        for key_ in ["action", "reward", "pscore"]:
            if key_ not in self.bandit_feedback:
                raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")
        self.bandit_feedback["action_by_behavior_policy"] = self.bandit_feedback[
            "action"
        ]
        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator

    def _create_estimator_inputs(
        self,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Create input dictionary to estimate policy value by subclasses of `BaseOffPolicyEstimator`"""
        if (
            not isinstance(action_by_evaluation_policy, np.ndarray)
            or action_by_evaluation_policy.ndim != 1
        ):
            raise ValueError(
                "action_by_evaluation_policy must be 1-dimensional ndarray"
            )
        if estimated_rewards_by_reg_model is None:
            logger.warning(
                "`estimated_rewards_by_reg_model` is not given; model dependent estimators such as DM or DR cannot be used."
            )
        elif isinstance(estimated_rewards_by_reg_model, dict):
            for estimator_name, value in estimated_rewards_by_reg_model.items():
                if not isinstance(value, np.ndarray):
                    raise ValueError(
                        f"estimated_rewards_by_reg_model[{estimator_name}] must be ndarray"
                    )
                elif value.shape != action_by_evaluation_policy.shape:
                    raise ValueError(
                        f"estimated_rewards_by_reg_model[{estimator_name}].shape and action_by_evaluation_policy.shape must be the same"
                    )
        elif estimated_rewards_by_reg_model.shape != action_by_evaluation_policy.shape:
            raise ValueError(
                "estimated_rewards_by_reg_model.shape and action_by_evaluation_policy.shape must be the same"
            )
        estimator_inputs = {
            estimator_name: {
                input_: self.bandit_feedback[input_]
                for input_ in ["reward", "action_by_behavior_policy", "pscore"]
            }
            for estimator_name in self.ope_estimators_
        }

        for estimator_name in self.ope_estimators_:
            estimator_inputs[estimator_name][
                "action_by_evaluation_policy"
            ] = action_by_evaluation_policy
            if isinstance(estimated_rewards_by_reg_model, dict):
                if estimator_name in estimated_rewards_by_reg_model:
                    estimator_inputs[estimator_name][
                        "estimated_rewards_by_reg_model"
                    ] = estimated_rewards_by_reg_model[estimator_name]
                else:
                    estimator_inputs[estimator_name][
                        "estimated_rewards_by_reg_model"
                    ] = None
            else:
                estimator_inputs[estimator_name][
                    "estimated_rewards_by_reg_model"
                ] = estimated_rewards_by_reg_model

        return estimator_inputs

    def estimate_policy_values(
        self,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
    ) -> Dict[str, float]:
        """Estimate policy value of evaluation policy.

        Parameters
        ------------
        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by evaluation policy, i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds,) or Dict[str, array-like], default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of a estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        Returns
        ----------
        policy_value_dict: Dict[str, float]
            Dictionary containing estimated policy values by OPE estimators.

        """
        policy_value_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_dict[estimator_name] = estimator.estimate_policy_value(
                **estimator_inputs[estimator_name]
            )

        return policy_value_dict

    def estimate_intervals(
        self,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate confidence intervals of policy values by nonparametric bootstrap procedure.

        Parameters
        ------------
        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by the (deterministic) evaluation policy, i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of a estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        policy_value_interval_dict: Dict[str, Dict[str, float]]
            Dictionary containing confidence intervals of estimated policy value estimated
            using nonparametric bootstrap procedure.

        """
        check_confidence_interval_arguments(
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        policy_value_interval_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_interval_dict[estimator_name] = estimator.estimate_interval(
                **estimator_inputs[estimator_name],
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

        return policy_value_interval_dict

    def summarize_off_policy_estimates(
        self,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Summarize policy values and their confidence intervals estimated by OPE estimators.

        Parameters
        ------------
        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by the (deterministic) evaluation policy, i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of a estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        (policy_value_df, policy_value_interval_df): Tuple[DataFrame, DataFrame]
            Policy values and their confidence intervals estimated by OPE estimators.

        """
        policy_value_df = DataFrame(
            self.estimate_policy_values(
                action_by_evaluation_policy=action_by_evaluation_policy,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            ),
            index=["estimated_policy_value"],
        )
        policy_value_interval_df = DataFrame(
            self.estimate_intervals(
                action_by_evaluation_policy=action_by_evaluation_policy,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )
        )
        policy_value_of_behavior_policy = self.bandit_feedback["reward"].mean()
        policy_value_df = policy_value_df.T
        if policy_value_of_behavior_policy <= 0:
            logger.warning(
                f"Policy value of the behavior policy is {policy_value_of_behavior_policy} (<=0); relative estimated policy value is set to np.nan"
            )
            policy_value_df["relative_estimated_policy_value"] = np.nan
        else:
            policy_value_df["relative_estimated_policy_value"] = (
                policy_value_df.estimated_policy_value / policy_value_of_behavior_policy
            )
        return policy_value_df, policy_value_interval_df.T

    def visualize_off_policy_estimates(
        self,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize policy values estimated by OPE estimators.

        Parameters
        ----------
        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by the (deterministic) evaluation policy, i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of a estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        is_relative: bool, default=False,
            If True, the method visualizes the estimated policy values of evaluation policy
            relative to the ground-truth policy value of behavior policy.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If 'None' is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "fig_dir must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "fig_dir must be a string"

        estimated_round_rewards_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_round_rewards_dict[
                estimator_name
            ] = estimator._estimate_round_rewards(**estimator_inputs[estimator_name])
        estimated_round_rewards_df = DataFrame(estimated_round_rewards_dict)
        estimated_round_rewards_df.rename(
            columns={key: key.upper() for key in estimated_round_rewards_dict.keys()},
            inplace=True,
        )
        if is_relative:
            estimated_round_rewards_df /= self.bandit_feedback["reward"].mean()

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=estimated_round_rewards_df,
            ax=ax,
            ci=100 * (1 - alpha),
            n_boot=n_bootstrap_samples,
            seed=random_state,
        )
        plt.xlabel("OPE Estimators", fontsize=25)
        plt.ylabel(
            f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=20
        )
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=25 - 2 * len(self.ope_estimators))

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def evaluate_performance_of_estimators(
        self,
        ground_truth_policy_value: float,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        metric: str = "relative-ee",
    ) -> Dict[str, float]:
        """Evaluate estimation performance of OPE estimators.

        Note
        ------
        Evaluate the estimation performance of OPE estimators by relative estimation error (relative-EE) or squared error (SE):

        .. math ::
            \\text{Relative-EE} (\\hat{V}; \\mathcal{D}) = \\left|  \\frac{\\hat{V}(\\pi; \\mathcal{D}) - V(\\pi)}{V(\\pi)} \\right|,

        .. math ::
            \\text{SE} (\\hat{V}; \\mathcal{D}) = \\left(\\hat{V}(\\pi; \\mathcal{D}) - V(\\pi) \\right)^2,

        where :math:`V({\\pi})` is the ground-truth policy value of the evalation policy :math:`\\pi_e` (often estimated using on-policy estimation).
        :math:`\\hat{V}(\\pi; \\mathcal{D})` is an estimated policy value by an OPE estimator :math:`\\hat{V}` and logged bandit feedback :math:`\\mathcal{D}`.

        Parameters
        ----------
        ground_truth policy value: float
            Ground_truth policy value of evaluation policy, i.e., :math:`V(\\pi)`.
            With Open Bandit Dataset, we use an on-policy estimate of the policy value as its ground-truth.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by the (deterministic) evaluation policy, i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of a estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        metric: str, default="relative-ee"
            Evaluation metric to evaluate and compare the estimation performance of OPE estimators.
            Must be "relative-ee" or "se".

        Returns
        ----------
        eval_metric_ope_dict: Dict[str, float]
            Dictionary containing evaluation metric for evaluating the estimation performance of OPE estimators.

        """

        if not isinstance(ground_truth_policy_value, float):
            raise ValueError(
                f"ground_truth_policy_value must be a float, but {ground_truth_policy_value} is given"
            )
        if metric not in ["relative-ee", "se"]:
            raise ValueError(
                f"metric must be either 'relative-ee' or 'se', but {metric} is given"
            )
        if metric == "relative-ee" and ground_truth_policy_value == 0.0:
            raise ValueError(
                "ground_truth_policy_value must be non-zero when metric is relative-ee"
            )

        eval_metric_ope_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_policy_value = estimator.estimate_policy_value(
                **estimator_inputs[estimator_name]
            )
            if metric == "relative-ee":
                relative_ee_ = estimated_policy_value - ground_truth_policy_value
                relative_ee_ /= ground_truth_policy_value
                eval_metric_ope_dict[estimator_name] = np.abs(relative_ee_)
            elif metric == "se":
                se_ = (estimated_policy_value - ground_truth_policy_value) ** 2
                eval_metric_ope_dict[estimator_name] = se_
        return eval_metric_ope_dict

    def summarize_estimators_comparison(
        self,
        ground_truth_policy_value: float,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        metric: str = "relative-ee",
    ) -> DataFrame:
        """Summarize performance comparison of OPE estimators.

        Parameters
        ----------
        ground_truth policy value: float
            Ground_truth policy value of evaluation policy, i.e., :math:`V(\\pi)`.
            With Open Bandit Dataset, we use an on-policy estimate of the policy value as ground-truth.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by the (deterministic) evaluation policy, i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        metric: str, default="relative-ee"
            Evaluation metric to evaluate and compare the estimation performance of OPE estimators.
            Must be either "relative-ee" or "se".

        Returns
        ----------
        eval_metric_ope_df: DataFrame
            Evaluation metric to evaluate and compare the estimation performance of OPE estimators.

        """
        eval_metric_ope_df = DataFrame(
            self.evaluate_performance_of_estimators(
                ground_truth_policy_value=ground_truth_policy_value,
                action_by_evaluation_policy=action_by_evaluation_policy,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                metric=metric,
            ),
            index=[metric],
        )
        return eval_metric_ope_df.T

    def visualize_off_policy_estimates_of_multiple_policies(
        self,
        policy_name_list: List[str],
        action_by_evaluation_policy_list: List[np.ndarray],
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize policy values estimated by OPE estimators.

        Parameters
        ----------
        policy_name_list: List[str]
            List of the names of evaluation policies.

        action_by_evaluation_policy_list: List[array-like, shape (n_rounds, n_actions, len_list)]
            List of action values given by the (deterministic) evaluation policies, i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of an estimator as a key, the corresponding value is used.
            When it is not given, model-dependent estimators such as DM and DR cannot be used.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        is_relative: bool, default=False,
            If True, the method visualizes the estimated policy values of evaluation policy
            relative to the ground-truth policy value of behavior policy.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If 'None' is given, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        if len(policy_name_list) != len(action_by_evaluation_policy_list):
            raise ValueError(
                "the length of policy_name_list must be the same as action_by_evaluation_policy_list"
            )
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "fig_dir must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "fig_dir must be a string"

        estimated_round_rewards_dict = {
            estimator_name: {} for estimator_name in self.ope_estimators_
        }

        for policy_name, action_by_evaluation_policy in zip(
            policy_name_list, action_by_evaluation_policy_list
        ):
            estimator_inputs = self._create_estimator_inputs(
                action_by_evaluation_policy=action_by_evaluation_policy,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )
            for estimator_name, estimator in self.ope_estimators_.items():
                estimated_round_rewards_dict[estimator_name][
                    policy_name
                ] = estimator._estimate_round_rewards(
                    **estimator_inputs[estimator_name]
                )

        plt.style.use("ggplot")
        fig = plt.figure(figsize=(8, 6.2 * len(self.ope_estimators_)))

        for i, estimator_name in enumerate(self.ope_estimators_):
            estimated_round_rewards_df = DataFrame(
                estimated_round_rewards_dict[estimator_name]
            )
            if is_relative:
                estimated_round_rewards_df /= self.bandit_feedback["reward"].mean()

            ax = fig.add_subplot(len(action_by_evaluation_policy_list), 1, i + 1)
            sns.barplot(
                data=estimated_round_rewards_df,
                ax=ax,
                ci=100 * (1 - alpha),
                n_boot=n_bootstrap_samples,
                seed=random_state,
            )
            ax.set_title(estimator_name.upper(), fontsize=20)
            ax.set_ylabel(
                f"Estimated Policy Value (± {np.int(100*(1 - alpha))}% CI)", fontsize=20
            )
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=25 - 2 * len(policy_name_list))

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))
