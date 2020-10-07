# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..utils import estimate_confidence_interval_by_bootstrap


@dataclass
class BaseOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for off-policy estimators."""

    @abstractmethod
    def _estimate_round_rewards(self) -> np.ndarray:
        """Estimate rewards for each round."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate policy value of a evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure."""
        raise NotImplementedError


@dataclass
class ReplayMethod(BaseOffPolicyEstimator):
    """Estimate the policy value by Relpay Method (RM).

    Replay Method (RM) estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{RM} (\\pi_e; \\mathcal{D}) =
        \\frac{\\sum_{t=1}^T  Y_t \\mathbb{I} \\{ \\pi_e (x_t) = a_t \\}}{\\sum_{t=1}^T \\mathbb{I} \\{\\pi_e (x_t) = a_t \\}}

    where :math:`\\pi_e: \\mathcal{X} \\rightarrow \\mathcal{A}` is the function
    representing the action choices of the evaluation policy during the offline bandit simulation.

    Parameters
    ----------
    estimator_name: str, default: 'rm'.
        Name of off-policy estimator.

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
        position: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds, )
            Rewards estimated by the Replay Method for each round.

        """
        action_match = np.array(
            action_dist[np.arange(action.shape[0]), action, position] == 1
        )
        round_rewards = np.zeros_like(action_match)
        if action_match.sum() > 0.0:
            round_rewards = action_match * reward / action_match.mean()
        return round_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of a evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t = Y(a_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given counterfactual or evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward, action=action, position=position, action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        action_dist: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        alpha: float, default: 0.05
            P-value.

        n_bootstrap_samples: int, default: 10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default: None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward, action=action, position=position, action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class InverseProbabilityWeighting(BaseOffPolicyEstimator):
    """Estimate the policy value by Inverse Probability Weighting (IPW).

    Inverse Probability Weighting (IPW) estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{IPW} (\\pi_e; \\mathcal{D}) = \\frac{1}{T} \\sum_{t=1}^T Y_t \\frac{\\pi_e (A_t | X_t)}{\\pi_b (A_t | X_t)}

    where :math:`\\mathcal{D}=\\{ (X_t,A_t,Y_t) \\}_{t=1}^{T}` is logged bandit feedback data collected by :math:`\\pi_b`.

    IPW re-weights the rewards s by the ratio of the evaluation policy and behavior policy (importance weight).
    When the behavior policy is known, the IPW estimator is unbiased and consistent for the policy value.
    However, it can have a large variance, especially when the evaluation policy significantly deviates from the behavior policy.

    Parameters
    ------------
    min_iw: float, default: 0.
        Minimum value of importance weight.
        Importance weights larger than this parameter would be clipped.

    estimator_name: str, default: 'ipw'.
        Name of off-policy estimator.

    References
    ------------
    Alex Strehl, John Langford, Lihong Li, and Sham M Kakade.
    "Learning from Logged Implicit Exploration Data"., 2010.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    min_iw: float = 0.0
    estimator_name: str = "ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert (
            self.min_iw >= 0.0
        ), f"minimum propensity score must be larger than or equal to zero, but {self.min_iw} is given"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds, )
            Rewards estimated by the Replay Method for each round.

        """
        importance_weight = (
            action_dist[np.arange(action.shape[0]), action, position] / pscore
        )
        return reward * np.maximum(importance_weight, self.min_iw)

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of a evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given counterfactual or evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position
            by the evaluation policy (can be deterministic).

        alpha: float, default: 0.05
            P-value.

        n_bootstrap_samples: int, default: 10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default: None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
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
class SelfNormalizedInverseProbabilityWeighting(InverseProbabilityWeighting):
    """Estimate the policy value by Self-Normalized Inverse Probability Weighting (SNIPW).

    Self-Normalized Inverse Probability Weighting (SNIPW) estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{SNIPW} (\\pi_e; \\mathcal{D}) =
        \\frac{\\sum_{t=1}^T Y_t \\frac{\\pi_e (A_t | X_t)}{\\pi_b (A_t | X_t)}}{\\sum_{t=1}^T \\frac{\\pi_e (A_t | X_t)}{\\pi_b (A_t | X_t)}}

    where :math:`\\mathcal{D}=\\{ (X_t,A_t,Y_t) \\}_{t=1}^{T}` is logged bandit feedback data collected by :math:`\\pi_b`.

    SNIPW re-weights the observed rewards by the self-normalized importance weihgt.
    This estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the policy value and increases the stability in some senses.
    See the references for the detailed discussions.

    Parameters
    ----------
    min_iw: float, default: 0.
        Minimum value of importance weight.
        Importance weights larger than this parameter would be clipped.

    estimator_name: str, default: 'snipw'.
        Name of off-policy estimator.

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
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds, )
            Rewards estimated by the SNIPW estimator for each round.

        """
        importance_weight = (
            action_dist[np.arange(action.shape[0]), action, position] / pscore
        )
        return reward * importance_weight / importance_weight.mean()


@dataclass
class DirectMethod(BaseOffPolicyEstimator):
    """Estimate the policy value by Direct Method (DM).

    DM first learns a supervised machine learning model, such as random forest, ridge regression, and gradient boosting,
    to estimate the mean reward function (:math:`\\mu(x, a) = E[Y(a) | X=x]`).
    It then uses it to estimate the policy value as follows.

    .. math::

        \\hat{V}_{DM} (\\pi_e; \\mathcal{D}, \\hat{\\mu}) = \\frac{1}{T} \\sum_{t=1}^T \\sum_{a \\in \\mathcal{A}} \\hat{\\mu} (X_t, a) \\pi(a | X_t)

    where :math:`\\mathcal{D}=\\{ (X_t,A_t,Y_t) \\}_{t=1}^{T}` is logged bandit feedback data collected by :math:`\\pi_b`.
    :math:`\\hat{\\mu} (x, a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`, which supports several fitting methods specific to OPE.

    If the regression model is a good approximation to the mean reward function,
    this estimator accurately estimates the policy value of the evaluation policy.
    If the regression function fails to approximate the mean reward function well,
    however, the final estimator is no longer consistent.

    Parameters
    ----------
    estimator_name: str, default: 'dm'.
        Name of off-policy estimator.

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
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of a evaluation policy.

        Parameters
        ----------
        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{\\mu}`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds, )
            Rewards estimated by the DM estimator for each round.

        """
        n_rounds = position.shape[0]
        estimated_rewards_by_reg_model_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        action_dist_at_position = action_dist[np.arange(n_rounds), :, position]
        return np.average(
            estimated_rewards_by_reg_model_at_position,
            weights=action_dist_at_position,
            axis=1,
        )

    def estimate_policy_value(
        self,
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of a evaluation policy.

        Parameters
        ----------
        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{\\mu}`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given counterfactual or evaluation policy.

        """
        return self._estimate_round_rewards(
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{\\mu}`.

        alpha: float, default: 0.05
            P-value.

        n_bootstrap_samples: int, default: 10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default: None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
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
class DoublyRobust(InverseProbabilityWeighting):
    """Estimate the policy value by Doubly Robust (DR).

    Similar to DM, DR first learns a supervised machine learning model, such as random forest, ridge regression, and gradient boosting,
    to estimate the mean reward function (:math:`\\mu(x, a) = E[Y(a) | X=x]`).
    It then uses it to estimate the policy value as follows.

    .. math::

            \\hat{V}_{DR} (\\pi_e; \\mathcal{D}, \\hat{\\mu}) =
            \\hat{V}_{DM} (\\pi_e; \\mathcal{D}, \\hat{\\mu})
            + \\frac{1}{T} \\sum_{t=1}^T (Y_t - \\hat{\\mu} (X_t, A_t)) \\frac{\\pi_e (A_t | X_t)}{\\pi_b (A_t | X_t)}

    where :math:`\\mathcal{D}=\\{ (X_t,A_t,Y_t) \\}_{t=1}^{T}` is logged bandit feedback data collected by :math:`\\pi_b`.
    :math:`\\hat{\\mu} (x, a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    DR mimics IPW to use a weighted version of rewards, but DR also uses the estimated mean reward
    function (the regression model) as a control variate to decrease the variance.
    It preserves the consistency of IPW if either the importance weight or
    the mean reward estimator is accurate (a property called double robustness).
    Moreover, DR is semiparametric efficient when the mean reward estimator is correctly specified.

    Parameters
    ----------
    min_iw: float, default: 0.
        Minimum value of importance weight.
        Importance weights larger than this parameter would be clipped.

    estimator_name: str, default: 'dr'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    """

    min_iw: float = 0.0
    estimator_name: str = "dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        super().__post_init__()

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{\\mu}`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds, )
            Rewards estimated by the DR estimator for each round.

        """
        n_rounds = position.shape[0]
        estimated_rewards_by_reg_model_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        action_dist_at_position = action_dist[np.arange(n_rounds), :, position]
        round_rewards = np.average(
            estimated_rewards_by_reg_model_at_position,
            weights=action_dist_at_position,
            axis=1,
        )
        importance_weight = (
            action_dist[np.arange(action.shape[0]), action, position] / pscore
        )
        estimated_observed_rewards = estimated_rewards_by_reg_model[
            np.arange(action.shape[0]), action, position
        ]
        round_rewards += np.maximum(importance_weight, self.min_iw) * (
            reward - estimated_observed_rewards
        )
        return round_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
    ) -> float:
        """Estimate policy value of a evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{\\mu}`.

        Returns
        ----------
        V_hat: float
            Estimated policy value by the DR estimator.

        """
        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{\\mu}`.

        alpha: float, default: 0.05
            P-value.

        n_bootstrap_samples: int, default: 10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default: None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        estimated_round_rewards = self._estimate_round_rewards(
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
class SelfNormalizedDoublyRobust(DoublyRobust):
    """Estimate the policy value by Self-Normalized Doubly Robust (SNDR).

    Self-Normalized Doubly Robust estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

            \\hat{V}_{SNDR} (\\pi_e; \\mathcal{D}, \\hat{\\mu}) =
            \\frac{T}{\\sum_{t=1}^T \\frac{\\pi_e (A_t | X_t)}{\\pi_b (A_t | X_t)}} \\hat{V}_{DR} (\\pi_e; \\mathcal{D}, \\hat{\\mu})

    where :math:`\\mathcal{D}=\\{ (X_t,A_t,Y_t) \\}_{t=1}^{T}` is logged bandit feedback data collected by :math:`\\pi_b`.
    :math:`\\hat{\\mu} (x, a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    Similar to Self-Normalized Inverse Probability Weighting, SNDR estimator applies the self-normalized importance weighting technique to
    increase the stability of the original Doubly Robust estimator.
    See also the description of `obp.ope.SelfNormalizedInverseProbabilityWeighting` for details.

    Parameters
    ----------
    estimator_name: str, default: 'sndr'.
        Name of off-policy estimator.

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
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{\\mu}`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds, )
            Rewards estimated by the SNDR estimator for each round.

        """
        n_rounds = position.shape[0]
        estimated_rewards_by_reg_model_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        action_dist_at_position = action_dist[np.arange(n_rounds), :, position]
        round_rewards = np.average(
            estimated_rewards_by_reg_model_at_position,
            weights=action_dist_at_position,
            axis=1,
        )
        importance_weight = (
            action_dist[np.arange(action.shape[0]), action, position] / pscore
        )
        estimated_observed_rewards = estimated_rewards_by_reg_model[
            np.arange(action.shape[0]), action, position
        ]
        round_rewards += importance_weight * (reward - estimated_observed_rewards)
        return round_rewards * importance_weight.mean()


@dataclass
class SwitchInverseProbabilityWeighting(DoublyRobust):
    """Estimate the policy value by Switch Inverse Probability Weighting (Switch-IPW).

    Switch Inverse Probability Weighting aims to reduce the variance of the Inverse Probability Weighting estimator by using the direct method insted
    when the importance weight is large. This estimator estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

            \\hat{V}_{Switch-IPW} (\\pi_e; \\mathcal{D}, \\hat{\\mu}, \\tau)
            =  \\frac{1}{T} \\sum_{t=1}^T \\sum_{a \\in \\mathcal{A}} \\hat{\\mu} (X_t, a) \\pi(a | X_t) \\mathbb{I} \\{ \\frac{\\pi_e (A_t | X_t)}{\\pi_b (A_t | X_t)} > \\tau \\}
            + \\frac{1}{T} \\sum_{t=1}^T \\frac{\\pi_e (A_t | X_t)}{\\pi_b (A_t | X_t)} \\mathbb{I} \\{ \\frac{\\pi_e (A_t | X_t)}{\\pi_b (A_t | X_t)} \\le \\tau \\}

    where :math:`\\mathcal{D}=\\{ (X_t,A_t,Y_t) \\}_{t=1}^{T}` is logged bandit feedback data collected by :math:`\\pi_b`.
    :math:`\\tau (\\ge 0)` is the *switching hyperparameter*, which decides the *threshold* for the importance weight.
    :math:`\\hat{\\mu} (x, a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    Parameters
    ----------
    tau: float, default: 1
        Switching hyperparameter. When the density ratio is larger than this parameter, the DM estimator is applied, otherwise the IPW estimator is applied.
        This hyperparameter should be larger than 1., otherwise it is meaningless.

    estimator_name: str, default: 'switch-ipw'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    """

    tau: float = 1
    estimator_name: str = "switch-ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert (
            self.tau >= 0.0
        ), f"switching hyperparameter should be larger than 1. but {self.tau} is given"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{\\mu}`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds, )
            Rewards estimated by the Switch-DR estimator for each round.

        """
        n_rounds = position.shape[0]
        importance_weight = (
            action_dist[np.arange(action.shape[0]), action, position] / pscore
        )
        switch_indicator = np.array(importance_weight <= self.tau, dtype=int)
        estimated_rewards_by_reg_model_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        action_dist_at_position = action_dist[np.arange(n_rounds), :, position]
        round_rewards = np.average(
            estimated_rewards_by_reg_model_at_position,
            weights=action_dist_at_position,
            axis=1,
        )
        round_rewards *= 1 - switch_indicator
        round_rewards += switch_indicator * importance_weight * reward
        return round_rewards


@dataclass
class SwitchDoublyRobust(DoublyRobust):
    """Estimate the policy value by Switch Doubly Robust (Switch-DR).

    Switch Doubly Robust aims to reduce the variance of the Doubly Robust estimator by using the direct method insted of doubly robust
    when the importance weight is large. This estimator estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

            \\hat{V}_{Switch-DR} (\\pi_e; \\mathcal{D}, \\hat{\\mu}, \\tau)
            = \\hat{V}_{DM} (\\pi_e; \\mathcal{D}, \\hat{\\mu})
            + \\frac{1}{T} \\sum_{t=1}^T (Y_t - \\hat{\\mu} (X_t, A_t)) \\frac{\\pi_e (A_t | X_t)}{\\pi_b (A_t | X_t)} \\mathbb{I} \\{ \\frac{\\pi_e (A_t | X_t)}{\\pi_b (A_t | X_t)} \\le \\tau \\}

    where :math:`\\mathcal{D}=\\{ (X_t,A_t,Y_t) \\}_{t=1}^{T}` is logged bandit feedback data collected by :math:`\\pi_b`.
    :math:`\\tau (\\ge 0)` is the *switching hyperparameter*, which decides the *threshold* for the importance weight.
    :math:`\\hat{\\mu} (x, a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    Parameters
    ----------
    tau: float, default: 1
        Switching hyperparameter. When the density ratio is larger than this parameter, the DM estimator is applied, otherwise the DR estimator is applied.
        This hyperparameter should be larger than 0., otherwise it is meaningless.

    estimator_name: str, default: 'switch-dr'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudík.
    "Optimal and Adaptive Off-policy Evaluation in Contextual Bandits", 2016.

    """

    tau: float = 1
    estimator_name: str = "switch-dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert (
            self.tau >= 0.0
        ), f"switching hyperparameter must be larger than or equal to zero, but {self.tau} is given"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{\\mu}`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds, )
            Rewards estimated by the Switch-DR estimator for each round.

        """
        n_rounds = position.shape[0]
        estimated_rewards_by_reg_model_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        action_dist_at_position = action_dist[np.arange(n_rounds), :, position]
        round_rewards = np.average(
            estimated_rewards_by_reg_model_at_position,
            weights=action_dist_at_position,
            axis=1,
        )
        importance_weight = (
            action_dist[np.arange(action.shape[0]), action, position] / pscore
        )
        estimated_observed_rewards = estimated_rewards_by_reg_model[
            np.arange(action.shape[0]), action, position
        ]
        switch_indicator = np.array(importance_weight <= self.tau, dtype=int)
        round_rewards += (
            switch_indicator * importance_weight * (reward - estimated_observed_rewards)
        )
        return round_rewards


@dataclass
class DoublyRobustWithShrinkage(DoublyRobust):
    """Estimate the policy value by Doubly Robust with optimistic shrinkage (DRos).

    DR with shrinkage replaces the importance weight in the original DR estimator with a new weight mapping
    found by directly optimizing sharp bounds on the resulting MSE.

    .. math::

            \\hat{V}_{DRos} (\\pi_e; \\mathcal{D}, \\hat{\\mu}, \\lambda) =
            \\hat{V}_{DM} (\\pi_e; \\mathcal{D}, \\hat{\\mu})
            + \\frac{1}{T} \\sum_{t=1}^T w_{o,\\lambda} (X_t, A_t) (Y_t - \\hat{\\mu} (X_t, A_t))

    where :math:`\\mathcal{D}=\\{ (X_t,A_t,Y_t) \\}_{t=1}^{T}` is logged bandit feedback data collected by :math:`\\pi_b`.
    :math:`\\hat{\\mu} (x, a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.
    :math:`w_{o,\\lambda} (X_t, A_t)` is a new weight by the shrinkage technique which is defined as

    .. math::

        w_{o,\\lambda} (X_t, A_t) := \\frac{\\lambda}{w^2(X_t, A_t) + \\lambda} w(X_t, A_t)

    where :math:`\\lambda` is a hyperparameter and :math:`w(X_t, A_t) = \\pi_e(X_t, A_t) / \\pi_b(X_t, A_t)` is the importance weight.
    When :math:`\\lambda=0`, we have :math:`w_{o,\\lambda} (X_t, A_t)=0` corresponding to the DM estimator.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, the weights increase and in the limit become equal to
    the original importance weight, corresponding to the standard DR estimator.
    Note that there is the other kind of the shrinkage technique called *pessimistic shrinkage*.
    DR with pessimistic shrinkage can be achieved by controlling the clipping hyperparameter of the original DR estimator
    (i.e., obp.ope.DoublyRobust), and thus is not implemented in this class.


    Parameters
    ----------
    lambda_: float
        Shrinkage hyperparameter. This hyperparameter should be larger than 0., otherwise it is meaningless.

    estimator_name: str, default: 'dr-os'.
        Name of off-policy estimator.

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
        assert (
            self.lambda_ >= 0.0
        ), f"shrinkage hyperparameter must be larger than or equal to zero, but {self.lambda_} is given"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like, shape (n_rounds, )
            Observed reward (or outcome) for each round, i.e., :math:`Y_t=Y(A_t)`.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds, )
            Propensity score or the probability of an action being selected by behavior policy, i.e., :math:`\\pi_b(a|X_t=x)`.

        action_dist: array-like shape (n_rounds, n_actions, len_list)
            Distribution over actions, i.e., probability of items being selected at each position by the evaluation policy (can be deterministic).

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{\\mu}`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds, )
            Rewards estimated by the DR estimator for each round.

        """
        n_rounds = position.shape[0]
        estimated_rewards_by_reg_model_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        action_dist_at_position = action_dist[np.arange(n_rounds), :, position]
        round_rewards = np.average(
            estimated_rewards_by_reg_model_at_position,
            weights=action_dist_at_position,
            axis=1,
        )
        importance_weight = (
            action_dist[np.arange(action.shape[0]), action, position] / pscore
        )
        shrinkage_weight = (self.lambda_ * importance_weight) / (
            importance_weight ** 2 + self.lambda_
        )
        estimated_observed_rewards = estimated_rewards_by_reg_model[
            np.arange(action.shape[0]), action, position
        ]
        round_rewards += shrinkage_weight * (reward - estimated_observed_rewards)
        return round_rewards
