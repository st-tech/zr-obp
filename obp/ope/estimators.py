# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators."""
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar

from ..utils import check_array
from ..utils import check_ope_inputs
from ..utils import estimate_confidence_interval_by_bootstrap
from .helper import estimate_bias_in_ope
from .helper import estimate_high_probability_upper_bound_bias


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators."""

    @abstractmethod
    def _estimate_round_rewards(self) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap."""
        raise NotImplementedError


@dataclass
class ReplayMethod(BaseOffPolicyEstimator):
    """Relpay Method (RM).

    Note
    -------
    RM estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{RM}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{n}[\\mathbb{I} \\{ \\pi_e(x_t)=a_t \\} r_t]}{\\mathbb{E}_{n}[\\mathbb{I} \\{ \\pi_e(x_t)=a_t \\}]},

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`\\pi_e: \\mathcal{X} \\rightarrow \\mathcal{A}` is the function
    representing action choices by the evaluation policy realized during offline bandit simulation.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    Parameters
    ----------
    estimator_name: str, default='rm'.
        Name of the estimator.

    References
    ------------
    Lihong Li, Wei Chu, John Langford, and Xuanhui Wang.
    "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms.", 2011.

    """

    estimator_name: str = "rm"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        action_match = np.array(
            action_dist[np.arange(action.shape[0]), action, position] == 1
        )
        estimated_rewards = np.zeros_like(action_match)
        if action_match.sum() > 0.0:
            estimated_rewards = action_match * reward / action_match.mean()
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
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

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

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
        check_ope_inputs(
            action_dist=action_dist, position=position, action=action, reward=reward
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class InverseProbabilityWeighting(BaseOffPolicyEstimator):
    """Inverse Probability Weighting (IPW) Estimator.

    Note
    -------
    IPW estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{IPW}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [ w(x_i,a_i) r_i],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    When the clipping is applied, a large importance weight is clipped as :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
    where :math:`\\lambda (>0)` is a hyperparameter to specify a maximum allowed importance weight.

    IPW re-weights the rewards by the ratio of the evaluation policy and behavior policy (importance weight).
    When the behavior policy is known, IPW is unbiased and consistent for the true policy value.
    However, it can have a large variance, especially when the evaluation policy significantly deviates from the behavior policy.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='ipw'.
        Name of the estimator.

    References
    ------------
    Alex Strehl, John Langford, Lihong Li, and Sham M Kakade.
    "Learning from Logged Implicit Exploration Data"., 2010.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambda_: float = np.inf
    use_estimated_pscore: bool = False
    estimator_name: str = "ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        return reward * iw

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

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
        ).mean()

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

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = True,
        delta: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate the MSE score of a given clipping hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        use_bias_upper_bound: bool, default=True
            Whether to use bias upper bound in hyperparameter tuning.
            If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound based on Bernstein inequality.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given clipping hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n = reward.shape[0]
        # estimate the sample variance of IPW with clipping
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                position=position,
            )
        )
        sample_variance /= n

        # estimate the (high probability) upper bound of the bias of IPW with clipping
        iw = action_dist[np.arange(n), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward, iw=iw, iw_hat=np.minimum(iw, self.lambda_), delta=delta
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
            )
        estimated_mse_score = sample_variance + (bias_term**2)

        return estimated_mse_score


@dataclass
class SelfNormalizedInverseProbabilityWeighting(InverseProbabilityWeighting):
    """Self-Normalized Inverse Probability Weighting (SNIPW) Estimator.

    Note
    -------
    SNIPW estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{SNIPW}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{n} [w(x_i,a_i) r_i]}{ \\mathbb{E}_{n} [w(x_i,a_i)]},

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    SNIPW normalizes the observed rewards by the self-normalized importance weihgt.
    This estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and gains some stability in OPE.
    See the reference papers for more details.

    Parameters
    ----------
    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='snipw'.
        Name of the estimator.

    References
    ----------
    Adith Swaminathan and Thorsten Joachims.
    "The Self-normalized Estimator for Counterfactual Learning.", 2015.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "snipw"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        return reward * iw / iw.mean()


@dataclass
class DirectMethod(BaseOffPolicyEstimator):
    """Direct Method (DM).

    Note
    -------
    DM first trains a supervised ML model, such as ridge regression and gradient boosting,
    to estimate the reward function (:math:`q(x,a) = \\mathbb{E}[r|x,a]`).
    It then uses the estimated rewards to estimate the policy value as follows.

    .. math::

        \\hat{V}_{\\mathrm{DM}} (\\pi_e; \\mathcal{D}, \\hat{q})
        &:= \\mathbb{E}_{n} \\left[ \\sum_{a \\in \\mathcal{A}} \\hat{q} (x_i,a) \\pi_e(a|x_i) \\right],    \\\\
        & =  \\mathbb{E}_{n}[\\hat{q} (x_i,\\pi_e)],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is the estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_i,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the reward function, please use `obp.ope.regression_model.RegressionModel`, which supports several fitting methods specific to OPE (such as cross-fitting).

    If the regression model (:math:`\\hat{q}`) is a good approximation to the true mean reward function,
    this estimator accurately estimates the policy value of the evaluation policy.
    If the regression function fails to approximate the reward function well,
    however, the final estimator is no longer consistent.

    Parameters
    ----------
    estimator_name: str, default='dm'.
        Name of the estimator.

    References
    ----------
    Alina Beygelzimer and John Langford.
    "The offset tree for learning with partial labels.", 2009.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "dm"

    def _estimate_round_rewards(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        n = position.shape[0]
        q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n), :, position]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        return np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )

    def estimate_policy_value(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

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
        check_ope_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            position=position,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

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
        check_ope_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            position=position,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class DoublyRobust(BaseOffPolicyEstimator):
    """Doubly Robust (DR) Estimator.

    Note
    -------
    Similar to DM, DR estimates the reward function (:math:`q(x,a) = \\mathbb{E}[r|x,a]`).
    It then uses the estimated rewards to estimate the policy value as follows.

    .. math::

        \\hat{V}_{\\mathrm{DR}} (\\pi_e; \\mathcal{D}, \\hat{q})
        := \\mathbb{E}_{n}[\\hat{q}(x_i,\\pi_e) +  w(x_i,a_i) (r_i - \\hat{q}(x_i,a_i))],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is the estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_i,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    When the clipping is applied, a large importance weight is clipped as :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
    where :math:`\\lambda (>0)` is a hyperparameter to specify a maximum allowed importance weight.

    To estimate the reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    DR mimics IPW to use a weighted version of rewards, but DR also uses the estimated mean reward
    function (the regression model) as a control variate to decrease the variance.
    It preserves the consistency of IPW if either the importance weight or
    the mean reward estimator is accurate (a property called double robustness).
    Moreover, DR is semiparametric efficient when the mean reward estimator is correctly specified.

    Parameters
    ----------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.
        DoublyRobust with a finite positive `lambda_` corresponds to DR with Pessimistic Shrinkage of Su et al.(2020)
        or CAB-DR of Su et al.(2019).

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='dr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    Yi Su, Lequn Wang, Michele Santacatterina, and Thorsten Joachims.
    "CAB: Continuous Adaptive Blending Estimator for Policy Evaluation and Learning", 2019.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudík.
    "Doubly robust off-policy evaluation with shrinkage.", 2020.

    """

    lambda_: float = np.inf
    use_estimated_pscore: bool = False
    estimator_name: str = "dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        n = action.shape[0]
        iw = action_dist[np.arange(n), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)

        q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n), :, position]
        q_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action, position]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += iw * (reward - q_hat_factual)

        return estimated_rewards

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
        """Estimate the policy value of evaluation policy.

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

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

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

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = True,
        delta: float = 0.05,
    ) -> float:
        """Estimate the MSE score of a given clipping hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        use_bias_upper_bound: bool, default=True
            Whether to use bias upper bound in hyperparameter tuning.
            If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound based on Bernstein inequality.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given clipping hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n = reward.shape[0]
        # estimate the sample variance of DR with clipping
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n

        # estimate the (high probability) upper bound of the bias of DR with clipping
        iw = action_dist[np.arange(n), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
                delta=delta,
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=np.minimum(iw, self.lambda_),
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
            )
        estimated_mse_score = sample_variance + (bias_term**2)

        return estimated_mse_score


@dataclass
class SelfNormalizedDoublyRobust(DoublyRobust):
    """Self-Normalized Doubly Robust (SNDR) Estimator.

    Note
    -------
    SNDR estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{SNDR}} (\\pi_e; \\mathcal{D}, \\hat{q}) :=
        \\mathbb{E}_{n} \\left[\\hat{q}(x_i,\\pi_e) +  \\frac{w(x_i,a_i) (r_i - \\hat{q}(x_i,a_i))}{\\mathbb{E}_{n}[ w(x_i,a_i) ]} \\right],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is the estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_i,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the reward function, please use `obp.ope.regression_model.RegressionModel`.

    Similar to SNIPW, SNDR estimator applies the self-normalized importance weighting technique to gain some stability.

    Parameters
    ----------
    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='sndr'.
        Name of the estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "sndr"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        n = action.shape[0]
        iw = action_dist[np.arange(n), action, position] / pscore
        q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n), :, position]
        q_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action, position]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += iw * (reward - q_hat_factual) / iw.mean()

        return estimated_rewards


@dataclass
class SwitchDoublyRobust(DoublyRobust):
    """Switch Doubly Robust (Switch-DR) Estimator.

    Note
    -------
    Switch-DR aims to reduce the variance of the DR estimator by using direct method when the importance weight is large.
    This estimator estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{SwitchDR}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{n} [\\hat{q}(x_i,\\pi_e) +  w(x_i,a_i) (r_i - \\hat{q}(x_i,a_i)) \\mathbb{I} \\{ w(x_i,a_i) \\le \\lambda \\}],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`. :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    :math:`\\lambda (\\ge 0)` is a switching hyperparameter, which decides the threshold for the importance weight.
    :math:`\\hat{q} (x,a)` is the estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_i,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the reward function, please use `obp.ope.regression_model.RegressionModel`.

    Parameters
    ----------
    lambda_: float, default=np.inf
        Switching hyperparameter. When importance weight is larger than this parameter, DM is applied, otherwise DR is used.
        Should be larger than or equal to 0., otherwise it is meaningless.

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

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """

    lambda_: float = np.inf
    estimator_name: str = "switch-dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        n = action.shape[0]
        iw = action_dist[np.arange(n), action, position] / pscore
        switch_indicator = np.array(iw <= self.lambda_, dtype=int)
        q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n), :, position]
        q_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action, position]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += switch_indicator * iw * (reward - q_hat_factual)

        return estimated_rewards

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = False,
        delta: float = 0.05,
    ) -> float:
        """Estimate the MSE score of a given switching hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        use_bias_upper_bound: bool, default=True
            Whether to use bias upper bound in hyperparameter tuning.
            If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound based on Bernstein inequality.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given switching hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n = reward.shape[0]
        # estimate the sample variance of Switch-DR (Eq.(8) of Wang et al.(2017))
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n

        # estimate the (high probability) upper bound of the bias of Switch-DR
        iw = action_dist[np.arange(n), action, position] / pscore
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward,
                iw=iw,
                iw_hat=iw * np.array(iw <= self.lambda_, dtype=int),
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
                delta=delta,
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=iw * np.array(iw <= self.lambda_, dtype=int),
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
            )
        estimated_mse_score = sample_variance + (bias_term**2)

        return estimated_mse_score


@dataclass
class DoublyRobustWithShrinkage(DoublyRobust):
    """Doubly Robust with optimistic shrinkage (DRos) Estimator.

    Note
    ------
    DRos shrinks the importance weight in the vanilla DR by directly optimizing sharp bounds on the resulting MSE.

    .. math::

        \\hat{V}_{\\mathrm{DRos}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{n} [\\hat{q}(x_i,\\pi_e) +  \\frac{\\lambda w(x_i,a_i)}{w^2(x_i,a_i) + \\lambda} w(x_i,a_i) (r_i - \\hat{q}(x_i,a_i))],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by
    behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_i,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is the estimated expected reward given :math:`x` and :math:`a`.
    To estimate the reward function, please use `obp.ope.regression_model.RegressionModel`.

    When :math:`\\lambda=0`, we have :math:`\\hat{w} (x,a;\\lambda)=0` corresponding to DM.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, :math:`\\hat{w} (x,a;\\lambda)` increases and in the limit becomes equal to the original importance weight, corresponding to the standard DR estimator.

    Parameters
    ----------
    lambda_: float
        Hyperparameter to shrink the importance weights. Should be larger than or equal to 0., otherwise it is meaningless.

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

    lambda_: float = 0.0
    estimator_name: str = "dr-os"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        n = action.shape[0]
        iw = action_dist[np.arange(n), action, position] / pscore
        if self.lambda_ < np.inf:
            iw_hat = (self.lambda_ * iw) / (iw**2 + self.lambda_)
        else:
            iw_hat = iw

        q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n), :, position]
        q_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action, position]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += iw_hat * (reward - q_hat_factual)

        return estimated_rewards

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = False,
        delta: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate the MSE score of a given shrinkage hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.

        use_bias_upper_bound: bool, default=True
            Whether to use bias upper bound in hyperparameter tuning.
            If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound based on Bernstein inequality.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given shrinkage hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n = reward.shape[0]
        # estimate the sample variance of DRos
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n

        # estimate the (high probability) upper bound of the bias of DRos
        iw = action_dist[np.arange(n), action, position] / pscore
        if self.lambda_ < np.inf:
            iw_hat = (self.lambda_ * iw) / (iw**2 + self.lambda_)
        else:
            iw_hat = iw
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward,
                iw=iw,
                iw_hat=iw_hat,
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
                delta=delta,
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=iw_hat,
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
            )
        estimated_mse_score = sample_variance + (bias_term**2)

        return estimated_mse_score


@dataclass
class SubGaussianInverseProbabilityWeighting(InverseProbabilityWeighting):
    """Sub-Gaussian Inverse Probability Weighting (SG-IPW) Estimator.

    Note
    ------
    Sub-Gaussian IPW replaces the importance weights in the vanilla IPW by applying the power mean as follows.

    .. math::

        \\hat{V}_{\\mathrm{SGIPW}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{n} [\\frac{w(x_i,a_i)}{1 - \\lambda + \\lambda \cdot w(x_i,a_i)} r_i ],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations
    collected by behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the true importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    Parameters
    ----------
    lambda_: float
        Hyperparameter to shrink the importance weights. Should be within the range of [0.0, 1.0].
        When `lambda_=0`, the estimator is identical to the vanilla DR.
        When `lambda_=1`, the importance weights will be uniform.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='sg-ipw'.
        Name of the estimator.

    References
    ----------
    Alberto Maria Metelli, Alessio Russo, and Marcello Restelli.
    "Subgaussian and Differentiable Importance Sampling for Off-Policy Evaluation and Learning.", 2021.

    """

    lambda_: float = 0.0
    estimator_name: str = "sg-ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
            max_val=1.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like or Tensor, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        iw_hat = iw / (1 - self.lambda_ + self.lambda_ * iw)
        estimated_rewards = iw_hat * reward

        return estimated_rewards

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = False,
        delta: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate the MSE score of a given shrinkage hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.

        use_bias_upper_bound: bool, default=True
            Whether to use bias upper bound in hyperparameter tuning.
            If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound based on Bernstein inequality.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given shrinkage hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n = reward.shape[0]
        # estimate the sample variance of DRos
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                position=position,
            )
        )
        sample_variance /= n

        # estimate the (high probability) upper bound of the bias of SGIPW
        iw = action_dist[np.arange(n), action, position] / pscore
        iw_hat = iw / (1 - self.lambda_ + self.lambda_ * iw)
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward,
                iw=iw,
                iw_hat=iw_hat,
                delta=delta,
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=iw_hat,
            )
        estimated_mse_score = sample_variance + (bias_term**2)

        return estimated_mse_score


@dataclass
class SubGaussianDoublyRobust(DoublyRobust):
    """Sub-Gaussian Doubly Robust (SG-DR) Estimator.

    Note
    ------
    Sub-Gaussian DR replaces the importance weights in the vanilla DR by applying the power mean as follows.

    .. math::

        \\hat{V}_{\\mathrm{SGDR}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{n} [\\hat{q}(x_i,\\pi_e) + \\frac{w(x_i,a_i)}{1 - \\lambda + \\lambda \cdot w(x_i,a_i)} (r_i - \\hat{q}(x_i,a_i))],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by behavior policy :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the true importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_i,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is the estimated expected reward given :math:`x` and :math:`a`.
    To estimate the reward function, please use `obp.ope.regression_model.RegressionModel`.

    Parameters
    ----------
    lambda_: float
        Hyperparameter to shrink the importance weights. Should be within the range of [0.0, 1.0].
        When `lambda_=0`, the estimator is identical to the vanilla DR.
        When `lambda_=1`, the importance weights will be uniform.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='sg-dr'.
        Name of the estimator.

    References
    ----------
    Alberto Maria Metelli, Alessio Russo, and Marcello Restelli.
    "Subgaussian and Differentiable Importance Sampling for Off-Policy Evaluation and Learning.", 2021.

    """

    lambda_: float = 0.0
    estimator_name: str = "sg-dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
            max_val=1.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like or Tensor, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        n = action.shape[0]
        iw = action_dist[np.arange(n), action, position] / pscore
        iw_hat = iw / (1 - self.lambda_ + self.lambda_ * iw)

        q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n), :, position]
        q_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action, position]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += iw_hat * (reward - q_hat_factual)

        return estimated_rewards

    def _estimate_mse_score(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        use_bias_upper_bound: bool = False,
        delta: float = 0.05,
        **kwargs,
    ) -> float:
        """Estimate the MSE score of a given shrinkage hyperparameter to conduct hyperparameter tuning.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.

        use_bias_upper_bound: bool, default=True
            Whether to use bias upper bound in hyperparameter tuning.
            If False, the direct bias estimator is used to estimate the MSE. See Su et al.(2020) for details.

        delta: float, default=0.05
            A confidence delta to construct a high probability upper bound based on Bernstein inequality.

        Returns
        ----------
        estimated_mse_score: float
            Estimated MSE score of a given shrinkage hyperparameter `lambda_`.
            MSE score is the sum of (high probability) upper bound of bias and the sample variance.
            This is estimated using the automatic hyperparameter tuning procedure
            based on Section 5 of Su et al.(2020).

        """
        n = reward.shape[0]
        # estimate the sample variance of DRos
        sample_variance = np.var(
            self._estimate_round_rewards(
                reward=reward,
                action=action,
                pscore=pscore,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                position=position,
            )
        )
        sample_variance /= n

        # estimate the (high probability) upper bound of the bias of SGDR
        iw = action_dist[np.arange(n), action, position] / pscore
        iw_hat = iw / (1 - self.lambda_ + self.lambda_ * iw)
        if use_bias_upper_bound:
            bias_term = estimate_high_probability_upper_bound_bias(
                reward=reward,
                iw=iw,
                iw_hat=iw_hat,
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
                delta=delta,
            )
        else:
            bias_term = estimate_bias_in_ope(
                reward=reward,
                iw=iw,
                iw_hat=iw_hat,
                q_hat=estimated_rewards_by_reg_model[np.arange(n), action, position],
            )
        estimated_mse_score = sample_variance + (bias_term**2)

        return estimated_mse_score


@dataclass
class BalancedInverseProbabilityWeighting(BaseOffPolicyEstimator):
    """Balanced Inverse Probability Weighting (B-IPW) Estimator.

    Note
    -------
    B-IPW estimates the policy value of evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{B-IPW}} (\\pi_e; \\mathcal{D}) := \\frac{\\mathbb{E}_{\\mathcal{D}} [\\hat{w}(x_i,a_i) r_i]}{\\mathbb{E}_{\\mathcal{D}} [\\hat{w}(x_i,a_i)},

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_t)\\}_{t=1}^{T}` is logged bandit data with :math:`n` observations collected by
    a behavior policy :math:`\\pi_b`.
    :math:`\\hat{w}(x,a):=\\Pr[C=1|x,a] / \\Pr[C=0|x,a]`, where :math:`\\Pr[C=1|x,a]` is the probability that the data coming from the evaluation policy given action :math:`a` and :math:`x`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    When the clipping is applied, large importance weights are clipped as :math:`\\hat{w_c}(x,a) := \\min \\{ \\lambda, \\hat{w}(x,a) \\}`
    where :math:`\\lambda (>0)` is a hyperparameter to define a maximum value allowed for importance weights.

    B-IPW re-weights the rewards by the importance weights estimated via a supervised classification procedure, and thus can be used even when the behavior policy (or the propensity score of the behavior policy) is not known. `obp.ope.ImportanceWeightEstimator` can be used to estimate the importance weights for B-IPW.


    Note that, in the reference paper, B-IPW is defined as follows (only when the evaluation policy is deterministic):

    .. math::

        \\hat{V}_{\\mathrm{B-IPW}} (\\pi_e; \\mathcal{D}) := \\frac{\\mathbb{E}_{\\mathcal{D}} [ \\hat{w}(x_t,\\pi_e (x_t)) r_i]}{\\mathbb{E}_{\\mathcal{D}} [ \\hat{w}(x_t,\\pi_e (x_t))},

    where :math:`\\pi_e` is a deterministic evaluation policy. We modify this original definition to adjust to stochastic evaluation policies.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    estimator_name: str, default='b-ipw'.
        Name of the estimator.

    References
    ------------
    Arjun Sondhi, David Arbour, and Drew Dimmery
    "Balanced Off-Policy Evaluation in General Action Spaces.", 2020.

    """

    lambda_: float = np.inf
    estimator_name: str = "b-ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        estimated_importance_weights: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        estimated_importance_weights: array-like, shape (n_rounds,)
            Importance weights estimated via supervised classification using `obp.ope.ImportanceWeightEstimator`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        iw = estimated_importance_weights
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        return reward * iw / iw.mean()

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_importance_weights: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        estimated_importance_weights: array-like, shape (n_rounds,)
            Importance weights estimated via supervised classification using `obp.ope.ImportanceWeightEstimator`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(
            array=estimated_importance_weights,
            name="estimated_importance_weights",
            expected_dim=1,
        )
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            estimated_importance_weights=estimated_importance_weights,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            estimated_importance_weights=estimated_importance_weights,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        estimated_importance_weights: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
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

        estimated_importance_weights: array-like, shape (n_rounds,)
            Importance weights estimated via supervised classification using `obp.ope.ImportanceWeightEstimator`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

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
        check_array(
            array=estimated_importance_weights,
            name="estimated_importance_weights",
            expected_dim=1,
        )
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            estimated_importance_weights=estimated_importance_weights,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            estimated_importance_weights=estimated_importance_weights,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
