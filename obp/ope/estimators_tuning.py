# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators with built-in hyperparameter tuning."""
from dataclasses import dataclass
from typing import Dict, Optional, Union, List

import numpy as np
from sklearn.utils import check_scalar

from .estimators import (
    InverseProbabilityWeighting,
    DoublyRobust,
    SwitchDoublyRobust,
    DoublyRobustWithShrinkage,
)
from ..utils import (
    estimate_confidence_interval_by_bootstrap,
    check_ope_inputs,
)


@dataclass
class InverseProbabilityWeightingTuning(InverseProbabilityWeighting):
    """Inverse Probability Weighting (IPW) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate clipping hyperparameters.
        The automatic hyperparameter tuning proposed by Wang et al.(2017) will choose the best hyperparameter value from the data.

    max_reward_value: int or float, default=None
        A maximum possible reward, which is necessary for the hyperparameter tuning.

    estimator_name: str, default='ipw'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambdas: List[float] = None
    max_reward_value: Optional[Union[int, float]] = None
    estimator_name = "ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        if isinstance(self.lambdas, list):
            if len(self.lambdas) == 0:
                raise ValueError("lambdas must not be empty")
            for lambda_ in self.lambdas:
                check_scalar(
                    lambda_,
                    name="an element of lambdas",
                    target_type=(int, float),
                    min_val=0.0,
                )
                if lambda_ != lambda_:
                    raise ValueError("an element of lambdas must not be nan")
        else:
            raise TypeError("lambdas must be a list")
        if self.max_reward_value is not None:
            check_scalar(
                self.max_reward_value,
                name="max_reward_value",
                target_type=(int, float),
            )

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        # tune the clipping hyperparameter
        self.estimated_mse_upper_bound_list = []
        for lambda_ in self.lambdas:
            estimated_mse_upper_bound = InverseProbabilityWeighting(
                lambda_=lambda_
            )._estimate_mse_upper_bound(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                max_reward_value=self.max_reward_value,
            )
            self.estimated_mse_upper_bound_list.append(estimated_mse_upper_bound)
        self.best_lambda_ = self.lambdas[np.argmin(self.estimated_mse_upper_bound_list)]

        return (
            InverseProbabilityWeighting(lambda_=self.best_lambda_)
            ._estimate_round_rewards(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
            )
            .mean()
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities
            by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        # tune the clipping hyperparameter
        self.estimated_mse_upper_bound_list = []
        for lambda_ in self.lambdas:
            estimated_mse_upper_bound = InverseProbabilityWeighting(
                lambda_=lambda_
            )._estimate_mse_upper_bound(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                max_reward_value=self.max_reward_value,
            )
            self.estimated_mse_upper_bound_list.append(estimated_mse_upper_bound)
        self.best_lambda_ = self.lambdas[np.argmin(self.estimated_mse_upper_bound_list)]

        estimated_round_rewards = InverseProbabilityWeighting(
            lambda_=self.best_lambda_
        )._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DoublyRobustTuning(DoublyRobust):
    """Doubly Robust (DR) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate clipping hyperparameters.
        The automatic hyperparameter tuning proposed by Wang et al.(2017) will choose the best hyperparameter value from the data.

    max_reward_value: int or float, default=None
            A maximum possible reward, which is necessary for the hyperparameter tuning.

    estimator_name: str, default='dr'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambdas: List[float] = None
    max_reward_value: Optional[Union[int, float]] = None
    estimator_name = "dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        if isinstance(self.lambdas, list):
            if len(self.lambdas) == 0:
                raise ValueError("lambdas must not be empty")
            for lambda_ in self.lambdas:
                check_scalar(
                    lambda_,
                    name="an element of lambdas",
                    target_type=(int, float),
                    min_val=0.0,
                )
                if lambda_ != lambda_:
                    raise ValueError("an element of lambdas must not be nan")
        else:
            raise TypeError("lambdas must be a list")
        if self.max_reward_value is not None:
            check_scalar(
                self.max_reward_value,
                name="max_reward_value",
                target_type=(int, float),
            )

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate policy value of an evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.

        Returns
        ----------
        V_hat: float
            Estimated policy value by the DR estimator.

        """
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        # tune the clipping hyperparameter
        self.estimated_mse_upper_bound_list = []
        for lambda_ in self.lambdas:
            estimated_mse_upper_bound = DoublyRobust(
                lambda_=lambda_
            )._estimate_mse_upper_bound(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                max_reward_value=self.max_reward_value,
            )
            self.estimated_mse_upper_bound_list.append(estimated_mse_upper_bound)
        self.best_lambda_ = self.lambdas[np.argmin(self.estimated_mse_upper_bound_list)]

        return (
            DoublyRobust(lambda_=self.best_lambda_)
            ._estimate_round_rewards(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )
            .mean()
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        # tune the clipping hyperparameter
        self.estimated_mse_upper_bound_list = []
        for lambda_ in self.lambdas:
            estimated_mse_upper_bound = DoublyRobust(
                lambda_=lambda_
            )._estimate_mse_upper_bound(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                max_reward_value=self.max_reward_value,
            )
            self.estimated_mse_upper_bound_list.append(estimated_mse_upper_bound)
        self.best_lambda_ = self.lambdas[np.argmin(self.estimated_mse_upper_bound_list)]

        estimated_round_rewards = DoublyRobust(
            lambda_=self.best_lambda_
        )._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SwitchDoublyRobustTuning(SwitchDoublyRobust):
    """Switch Doubly Robust (Switch-DR) with build-in hyperparameter tuning.

    Parameters
    ----------
    taus: List[float]
        A list of candidate switching hyperparameters.
        The automatic hyperparameter tuning proposed by Wang et al.(2017) will choose the best hyperparameter value from the data.

    max_reward_value: int or float, default=None
            A maximum possible reward, which is necessary for the hyperparameter tuning.
            If None is given, `reward.max()` is used.

    estimator_name: str, default='switch-dr'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    """

    taus: List[float] = None
    max_reward_value: Optional[float] = None
    estimator_name: str = "switch-dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        if isinstance(self.taus, list):
            if len(self.taus) == 0:
                raise ValueError("taus must not be empty")
            for tau in self.taus:
                check_scalar(
                    tau,
                    name="an element of taus",
                    target_type=(int, float),
                    min_val=0.0,
                )
                if tau != tau:
                    raise ValueError("an element of taus must not be nan")
        else:
            raise TypeError("taus must be a list")
        if self.max_reward_value is not None:
            check_scalar(
                self.max_reward_value,
                name="max_reward_value",
                target_type=(int, float),
            )

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate policy value of an evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.

        Returns
        ----------
        V_hat: float
            Estimated policy value by the DR estimator.

        """
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        # tune the switching hyperparameter
        self.estimated_mse_upper_bound_list = []
        for tau_ in self.taus:
            estimated_mse_upper_bound = SwitchDoublyRobust(
                tau=tau_
            )._estimate_mse_upper_bound(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                max_reward_value=self.max_reward_value,
            )
            self.estimated_mse_upper_bound_list.append(estimated_mse_upper_bound)
        self.best_tau = self.taus[np.argmin(self.estimated_mse_upper_bound_list)]

        return (
            SwitchDoublyRobust(tau=self.best_tau)
            ._estimate_round_rewards(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )
            .mean()
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        # tune the switching hyperparameter
        self.estimated_mse_upper_bound_list = []
        for tau_ in self.taus:
            estimated_mse_upper_bound = SwitchDoublyRobust(
                tau=tau_
            )._estimate_mse_upper_bound(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                max_reward_value=self.max_reward_value,
            )
            self.estimated_mse_upper_bound_list.append(estimated_mse_upper_bound)
        self.best_tau = self.taus[np.argmin(self.estimated_mse_upper_bound_list)]

        estimated_round_rewards = SwitchDoublyRobust(
            tau=self.best_tau
        )._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DoublyRobustWithShrinkageTuning(DoublyRobustWithShrinkage):
    """Doubly Robust with optimistic shrinkage (DRos) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate shrinkage hyperparameters.
        The automatic hyperparameter tuning proposed by Wang et al.(2017) will choose the best hyperparameter value from the data.

    max_reward_value: int or float, default=None
            A maximum possible reward, which is necessary for the hyperparameter tuning.

    estimator_name: str, default='dr-os'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambdas: List[float] = None
    max_reward_value: Optional[Union[int, float]] = None
    estimator_name = "dr-os"

    def __post_init__(self) -> None:
        """Initialize Class."""
        if isinstance(self.lambdas, list):
            if len(self.lambdas) == 0:
                raise ValueError("lambdas must not be empty")
            for lambda_ in self.lambdas:
                check_scalar(
                    lambda_,
                    name="an element of lambdas",
                    target_type=(int, float),
                    min_val=0.0,
                )
                if lambda_ != lambda_:
                    raise ValueError("an element of lambdas must not be nan")
        else:
            raise TypeError("lambdas must be a list")
        if self.max_reward_value is not None:
            check_scalar(
                self.max_reward_value,
                name="max_reward_value",
                target_type=(int, float),
            )

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate policy value of an evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.

        Returns
        ----------
        V_hat: float
            Estimated policy value by the DR estimator.

        """
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        # tune the shrinkage hyperparameter
        self.estimated_mse_upper_bound_list = []
        for lambda_ in self.lambdas:
            estimated_mse_upper_bound = DoublyRobustWithShrinkage(
                lambda_=lambda_
            )._estimate_mse_upper_bound(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                max_reward_value=self.max_reward_value,
            )
            self.estimated_mse_upper_bound_list.append(estimated_mse_upper_bound)
        self.best_lambda_ = self.lambdas[np.argmin(self.estimated_mse_upper_bound_list)]

        return (
            DoublyRobustWithShrinkage(lambda_=self.best_lambda_)
            ._estimate_round_rewards(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )
            .mean()
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        if not isinstance(estimated_rewards_by_reg_model, np.ndarray):
            raise ValueError("estimated_rewards_by_reg_model must be ndarray")
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(action, np.ndarray):
            raise ValueError("action must be ndarray")
        if not isinstance(pscore, np.ndarray):
            raise ValueError("pscore must be ndarray")

        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        # tune the shrinkage hyperparameter
        self.estimated_mse_upper_bound_list = []
        for lambda_ in self.lambdas:
            estimated_mse_upper_bound = DoublyRobustWithShrinkage(
                lambda_=lambda_
            )._estimate_mse_upper_bound(
                reward=reward,
                action=action,
                position=position,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                max_reward_value=self.max_reward_value,
            )
            self.estimated_mse_upper_bound_list.append(estimated_mse_upper_bound)
        self.best_lambda_ = self.lambdas[np.argmin(self.estimated_mse_upper_bound_list)]

        estimated_round_rewards = DoublyRobustWithShrinkage(
            lambda_=self.best_lambda_
        )._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
