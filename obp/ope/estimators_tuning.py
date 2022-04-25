# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators with built-in hyperparameter tuning."""
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar

from ..utils import check_array
from ..utils import check_ope_inputs
from .estimators import BaseOffPolicyEstimator
from .estimators import DoublyRobust
from .estimators import DoublyRobustWithShrinkage
from .estimators import InverseProbabilityWeighting
from .estimators import SubGaussianDoublyRobust
from .estimators import SubGaussianInverseProbabilityWeighting
from .estimators import SwitchDoublyRobust
from .helper import estimate_student_t_lower_bound


@dataclass
class BaseOffPolicyEstimatorTuning:
    """Base Class for Off-Policy Estimator with built-in hyperparameter tuning

    base_ope_estimator: BaseOffPolicyEstimator
        An OPE estimator with a hyperparameter
        (such as IPW/DR with clipping, Switch-DR, and DR with Shrinkage).

    lambdas: List[float]
        A list of candidate hyperparameter values.

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

    use_bias_upper_bound: bool, default=True
        Whether to use a bias upper bound in hyperparameter tuning.
        If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

    delta: float, default=0.1
        A confidence delta to construct a high probability upper bound used in SLOPE.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    Yi Su, Pavithra Srinath, and Akshay Krishnamurthy.
    "Adaptive Estimator Selection for Off-Policy Evaluation.", 2020.

    George Tucker and Jonathan Lee.
    "Improved Estimator Selection for Off-Policy Evaluation.", 2021.

    """

    base_ope_estimator: BaseOffPolicyEstimator = field(init=False)
    lambdas: List[float] = None
    tuning_method: str = "slope"
    use_bias_upper_bound: bool = True
    delta: float = 0.1
    use_estimated_pscore: bool = False

    def __new__(cls, *args, **kwargs):
        dataclass(cls)
        return super().__new__(cls)

    def _check_lambdas(self, min_val: float = 0.0, max_val: float = np.inf) -> None:
        """Check type and value of lambdas."""
        if isinstance(self.lambdas, list):
            if len(self.lambdas) == 0:
                raise ValueError("lambdas must not be empty")
            for hyperparam_ in self.lambdas:
                check_scalar(
                    hyperparam_,
                    name="an element of lambdas",
                    target_type=(int, float),
                    min_val=min_val,
                    max_val=max_val,
                )
                if hyperparam_ != hyperparam_:
                    raise ValueError("an element of lambdas must not be nan")
        else:
            raise TypeError("lambdas must be a list")

    def _check_init_inputs(self) -> None:
        """Initialize Class."""
        if self.tuning_method not in ["slope", "mse"]:
            raise ValueError(
                "`tuning_method` must be either 'slope' or 'mse'"
                f", but {self.tuning_method} is given"
            )
        if not isinstance(self.use_bias_upper_bound, bool):
            raise TypeError(
                "`use_bias_upper_bound` must be a bool"
                f", but {type(self.use_bias_upper_bound)} is given"
            )
        check_scalar(self.delta, "delta", (float), min_val=0.0, max_val=1.0)
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _tune_hyperparam_with_mse(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Find the best hyperparameter value from the candidate set by estimating the mse."""
        self.estimated_mse_score_dict = dict()
        for hyperparam_ in self.lambdas:
            estimated_mse_score = self.base_ope_estimator(
                lambda_=hyperparam_, use_estimated_pscore=self.use_estimated_pscore
            )._estimate_mse_score(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
                use_bias_upper_bound=self.use_bias_upper_bound,
                delta=self.delta,
            )
            self.estimated_mse_score_dict[hyperparam_] = estimated_mse_score
        return min(self.estimated_mse_score_dict.items(), key=lambda x: x[1])[0]

    def _tune_hyperparam_with_slope(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Find the best hyperparameter value from the candidate set by SLOPE."""
        C = np.sqrt(6) - 1
        theta_list_for_sort, cnf_list_for_sort = [], []
        for hyperparam_ in self.lambdas:
            estimated_round_rewards = self.base_ope_estimator(
                hyperparam_
            )._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
            theta_list_for_sort.append(estimated_round_rewards.mean())
            cnf = estimated_round_rewards.mean()
            cnf -= estimate_student_t_lower_bound(
                x=estimated_round_rewards,
                delta=self.delta,
            )
            cnf_list_for_sort.append(cnf)

        theta_list, cnf_list = [], []
        sorted_idx_list = np.argsort(cnf_list_for_sort)[::-1]
        for i, idx in enumerate(sorted_idx_list):
            cnf_i = cnf_list_for_sort[idx]
            theta_i = theta_list_for_sort[idx]
            if len(theta_list) < 1:
                theta_list.append(theta_i), cnf_list.append(cnf_i)
            else:
                theta_j, cnf_j = np.array(theta_list), np.array(cnf_list)
                if (np.abs(theta_j - theta_i) <= cnf_i + C * cnf_j).all():
                    theta_list.append(theta_i), cnf_list.append(cnf_i)
                else:
                    best_idx = sorted_idx_list[i - 1]
                    return self.lambdas[best_idx]

        return self.lambdas[sorted_idx_list[-1]]

    def estimate_policy_value_with_tuning(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        # tune hyperparameter if necessary
        if not hasattr(self, "best_hyperparam"):
            if self.tuning_method == "mse":
                self.best_hyperparam = self._tune_hyperparam_with_mse(
                    reward=reward,
                    action=action,
                    pscore=pscore_,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                )
            elif self.tuning_method == "slope":
                self.best_hyperparam = self._tune_hyperparam_with_slope(
                    reward=reward,
                    action=action,
                    pscore=pscore_,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                )

        return self.base_ope_estimator(
            lambda_=self.best_hyperparam, use_estimated_pscore=self.use_estimated_pscore
        ).estimate_policy_value(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            estimated_pscore=estimated_pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

    def estimate_interval_with_tuning(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        # tune hyperparameter if necessary
        if not hasattr(self, "best_hyperparam"):
            if self.tuning_method == "mse":
                self.best_hyperparam = self._tune_hyperparam_with_mse(
                    reward=reward,
                    action=action,
                    pscore=pscore_,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                )
            elif self.tuning_method == "slope":
                self.best_hyperparam = self._tune_hyperparam_with_slope(
                    reward=reward,
                    action=action,
                    pscore=pscore_,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                )

        return self.base_ope_estimator(self.best_hyperparam).estimate_interval(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            estimated_pscore=estimated_pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


class InverseProbabilityWeightingTuning(BaseOffPolicyEstimatorTuning):
    """Inverse Probability Weighting (IPW) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate clipping hyperparameters.
        The automatic hyperparameter tuning procedure proposed by Su et al.(2020)
        or Tucker and Lee.(2021) will choose the best hyperparameter value from the logged data.
        The candidate hyperparameter values will be sorted automatically to ensure the monotonicity
        assumption of SLOPE.

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

    use_bias_upper_bound: bool, default=True
        Whether to use a bias upper bound in hyperparameter tuning.
        If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

    delta: float, default=0.05
        A confidence delta to construct a high probability upper bound used in SLOPE.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='ipw'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    estimator_name: str = "ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = InverseProbabilityWeighting
        super()._check_lambdas()
        super()._check_init_inputs()
        self.lambdas.sort(reverse=True)

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
            estimated_pscore=estimated_pscore,
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities
            by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
           (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            estimated_pscore=estimated_pscore,
            action_dist=action_dist,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DoublyRobustTuning(BaseOffPolicyEstimatorTuning):
    """Doubly Robust (DR) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate clipping hyperparameters.
        The automatic hyperparameter tuning procedure proposed by Su et al.(2020)
        or Tucker and Lee.(2021) will choose the best hyperparameter value from the logged data.
        The candidate hyperparameter values will be sorted automatically to ensure the monotonicity
        assumption of SLOPE.

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambdas: List[float] = None
    estimator_name: str = "dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = DoublyRobust
        super()._check_lambdas()
        super()._check_init_inputs()
        self.lambdas.sort(reverse=True)

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            estimated_pscore=estimated_pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            estimated_pscore=estimated_pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SwitchDoublyRobustTuning(BaseOffPolicyEstimatorTuning):
    """Switch Doubly Robust (Switch-DR) with build-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate switching hyperparameters.
        The automatic hyperparameter tuning procedure proposed by Su et al.(2020)
        or Tucker and Lee.(2021) will choose the best hyperparameter value from the logged data.
        The candidate hyperparameter values will be sorted automatically to ensure the monotonicity
        assumption of SLOPE.

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='switch-dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    """

    estimator_name: str = "switch-dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = SwitchDoublyRobust
        super()._check_lambdas()
        super()._check_init_inputs()
        self.lambdas.sort(reverse=True)

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            estimated_pscore=estimated_pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            estimated_pscore=estimated_pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DoublyRobustWithShrinkageTuning(BaseOffPolicyEstimatorTuning):
    """Doubly Robust with optimistic shrinkage (DRos) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate shrinkage hyperparameters.
        The automatic hyperparameter tuning procedure proposed by Su et al.(2020)
        or Tucker and Lee.(2021) will choose the best hyperparameter value from the logged data.
        The candidate hyperparameter values will be sorted automatically to ensure the monotonicity
        assumption of SLOPE.

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='dr-os'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    estimator_name: str = "dr-os"

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = DoublyRobustWithShrinkage
        super()._check_lambdas()
        super()._check_init_inputs()
        self.lambdas.sort()

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            estimated_pscore=estimated_pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            estimated_pscore=estimated_pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


class SubGaussianInverseProbabilityWeightingTuning(BaseOffPolicyEstimatorTuning):
    """Sub-Gaussian Inverse Probability Weighting (SG-IPW) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate hyperparameter values, which should be in the range of [0.0, 1.0].
        The automatic hyperparameter tuning procedure proposed by Su et al.(2020)
        or Tucker and Lee.(2021) will choose the best hyperparameter value from the logged data.
        The candidate hyperparameter values will be sorted automatically to ensure the monotonicity
        assumption of SLOPE.

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

    use_bias_upper_bound: bool, default=True
        Whether to use a bias upper bound in hyperparameter tuning.
        If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

    delta: float, default=0.05
        A confidence delta to construct a high probability upper bound used in SLOPE.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='sg-ipw'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    Alberto Maria Metelli, Alessio Russo, and Marcello Restelli.
    "Subgaussian and Differentiable Importance Sampling for Off-Policy Evaluation and Learning.", 2021.

    """

    estimator_name: str = "sg-ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = SubGaussianInverseProbabilityWeighting
        super()._check_lambdas(max_val=1.0)
        super()._check_init_inputs()
        self.lambdas.sort()

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SubGaussianDoublyRobustTuning(BaseOffPolicyEstimatorTuning):
    """Sub-Gaussian Doubly Robust (SG-DR) with built-in hyperparameter tuning.

    Parameters
    ----------
    lambdas: List[float]
        A list of candidate hyperparameter values, which should be in the range of [0.0, 1.0].
        The automatic hyperparameter tuning procedure proposed by Su et al.(2020)
        or Tucker and Lee.(2021) will choose the best hyperparameter value from the logged data.
        The candidate hyperparameter values will be sorted automatically to ensure the monotonicity
        assumption of SLOPE.

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='sg-dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    Alberto Maria Metelli, Alessio Russo, and Marcello Restelli.
    "Subgaussian and Differentiable Importance Sampling for Off-Policy Evaluation and Learning.", 2021.

    """

    estimator_name: str = "sg-dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.base_ope_estimator = SubGaussianDoublyRobust
        super()._check_lambdas(max_val=1.0)
        super()._check_init_inputs()
        self.lambdas.sort()

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
