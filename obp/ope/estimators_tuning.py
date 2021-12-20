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
        Whether to use bias upper bound in hyperparameter tuning.
        If False, direct bias estimator is used to estimate the MSE.

    delta: float, default=0.05
        A confidence delta to construct a high probability upper bound based on the Bernstein’s inequality.

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
    delta: float = 0.05

    def __new__(cls, *args, **kwargs):
        dataclass(cls)
        return super().__new__(cls)

    def _check_lambdas(self) -> None:
        """Check type and value of lambdas."""
        if isinstance(self.lambdas, list):
            if len(self.lambdas) == 0:
                raise ValueError("lambdas must not be empty")
            for hyperparam_ in self.lambdas:
                check_scalar(
                    hyperparam_,
                    name="an element of lambdas",
                    target_type=(int, float),
                    min_val=0.0,
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
                hyperparam_
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
        theta_list, cnf_list = [], []
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
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        Returns
        ----------
        V_hat: float
            Policy value estimated by the DR estimator.

        """
        # tune hyperparameter if necessary
        if not hasattr(self, "best_hyperparam"):
            if self.tuning_method == "mse":
                self.best_hyperparam = self._tune_hyperparam_with_mse(
                    reward=reward,
                    action=action,
                    pscore=pscore,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                )
            elif self.tuning_method == "slope":
                self.best_hyperparam = self._tune_hyperparam_with_slope(
                    reward=reward,
                    action=action,
                    pscore=pscore,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                )

        return self.base_ope_estimator(self.best_hyperparam).estimate_policy_value(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

    def estimate_interval_with_tuning(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
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
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        # tune hyperparameter if necessary
        if not hasattr(self, "best_hyperparam"):
            if self.tuning_method == "mse":
                self.best_hyperparam = self._tune_hyperparam_with_mse(
                    reward=reward,
                    action=action,
                    pscore=pscore,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                )
            elif self.tuning_method == "slope":
                self.best_hyperparam = self._tune_hyperparam_with_slope(
                    reward=reward,
                    action=action,
                    pscore=pscore,
                    action_dist=action_dist,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                    position=position,
                )

        return self.base_ope_estimator(self.best_hyperparam).estimate_interval(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
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

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

    use_bias_upper_bound: bool, default=True
        Whether to use bias upper bound in hyperparameter tuning.
        If False, direct bias estimator is used to estimate the MSE.

    delta: float, default=0.05
        A confidence delta to construct a high probability upper bound based on the Bernstein’s inequality.

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

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
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
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities
            by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
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

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

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

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        Returns
        ----------
        V_hat: float
            Policy value estimated by the DR estimator.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=pscore, name="pscore", expected_dim=1)
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

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
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
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

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
        check_array(array=pscore, name="pscore", expected_dim=1)
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

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
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

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

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

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        Returns
        ----------
        V_hat: float
            Policy value estimated by the DR estimator.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=pscore, name="pscore", expected_dim=1)
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

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
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
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

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
        check_array(array=pscore, name="pscore", expected_dim=1)
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

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
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

    tuning_method: str, default="slope".
        A method used to tune the hyperparameter of an OPE estimator.
        Must be either of "slope" or "mse".
        Note that the implementation of "slope" is based on SLOPE++ proposed by Tucker and Lee.(2021),
        which improves the original SLOPE proposed by Su et al.(2020).

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

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """Estimate the policy value of evaluation policy with a tuned hyperparameter.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        Returns
        ----------
        V_hat: float
            Policy value estimated by the DR estimator.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=pscore, name="pscore", expected_dim=1)
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

        return super().estimate_policy_value_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
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
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

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
        check_array(array=pscore, name="pscore", expected_dim=1)
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

        return super().estimate_interval_with_tuning(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
