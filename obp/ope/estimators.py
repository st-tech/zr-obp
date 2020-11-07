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
        """Estimate policy value of an evaluation policy."""
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

        \\hat{V}_{\\mathrm{RM}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{\\mathcal{D}}[\\mathbb{I} \\{ \\pi_e (x_t) = a_t \\} r_t ]}{\\mathbb{E}_{\\mathcal{D}}[\\mathbb{I} \\{ \\pi_e (x_t) = a_t \\}]},

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by :math:`\\pi_b`.
    :math:`\\pi_e: \\mathcal{X} \\rightarrow \\mathcal{A}` is the function
    representing the deterministic action choices by the evaluation policy during the offline bandit simulation.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.

    Parameters
    ----------
    estimator_name: str, default='rm'.
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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the Replay Method for each round.

        """
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
        position: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ------------
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
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

        \\hat{V}_{\\mathrm{IPW}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{\\mathcal{D}} [ w(x_t,a_t) r_t],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.

    IPW re-weights the rewards by the ratio of the evaluation policy and behavior policy (importance weight).
    When the behavior policy is known, the IPW estimator is unbiased and consistent for the policy value.
    However, it can have a large variance, especially when the evaluation policy significantly deviates from the behavior policy.

    Parameters
    ------------
    estimator_name: str, default='ipw'.
        Name of off-policy estimator.

    References
    ------------
    Alex Strehl, John Langford, Lihong Li, and Sham M Kakade.
    "Learning from Logged Implicit Exploration Data"., 2010.

    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    estimator_name: str = "ipw"

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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by IPW for each round.

        """
        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        return reward * iw

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities
            by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

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

        \\hat{V}_{\\mathrm{SNIPW}} (\\pi_e; \\mathcal{D}) :=
        \\frac{\\mathbb{E}_{\\mathcal{D}} [w(x_t,a_t) r_t]}{ \\mathbb{E}_{\\mathcal{D}} [w(x_t,a_t)]},

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.

    SNIPW re-weights the observed rewards by the self-normalized importance weihgt.
    This estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the policy value and increases the stability in some senses.
    See the references for the detailed discussions.

    Parameters
    ----------
    estimator_name: str, default='snipw'.
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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the SNIPW estimator for each round.

        """
        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        return reward * iw / iw.mean()


@dataclass
class DirectMethod(BaseOffPolicyEstimator):
    """Estimate the policy value by Direct Method (DM).

    DM first learns a supervised machine learning model, such as ridge regression and gradient boosting,
    to estimate the mean reward function (:math:`q(x,a) = \\mathbb{E}[r|x,a]`).
    It then uses it to estimate the policy value as follows.

    .. math::

        \\hat{V}_{\\mathrm{DM}} (\\pi_e; \\mathcal{D}, \\hat{q})
        :=  \\mathbb{E}_{\\mathcal{D}}[\\hat{q} (x_t,\\pi_e)],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by :math:`\\pi_b`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`, which supports several fitting methods specific to OPE.

    If the regression model is a good approximation to the mean reward function,
    this estimator accurately estimates the policy value of the evaluation policy.
    If the regression function fails to approximate the mean reward function well,
    however, the final estimator is no longer consistent.

    Parameters
    ----------
    estimator_name: str, default='dm'.
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
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DM estimator for each round.

        """
        n_rounds = position.shape[0]
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        return np.average(q_hat_at_position, weights=pi_e_at_position, axis=1,)

    def estimate_policy_value(
        self,
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

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
        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

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

    Similar to DM, DR first learns a supervised machine learning model, such as ridge regression and gradient boosting,
    to estimate the mean reward function (:math:`q(x,a) = \\mathbb{E}[r|x,a]`).
    It then uses it to estimate the policy value as follows.

    .. math::

        \\hat{V}_{\\mathrm{DR}} (\\pi_e; \\mathcal{D}, \\hat{q})
        := \\mathbb{E}_{\\mathcal{D}}[\\hat{q}(x_t,\\pi_e) +  w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    DR mimics IPW to use a weighted version of rewards, but DR also uses the estimated mean reward
    function (the regression model) as a control variate to decrease the variance.
    It preserves the consistency of IPW if either the importance weight or
    the mean reward estimator is accurate (a property called double robustness).
    Moreover, DR is semiparametric efficient when the mean reward estimator is correctly specified.

    Parameters
    ----------
    estimator_name: str, default='dr'.
        Name of off-policy estimator.

    References
    ----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    """

    estimator_name: str = "dr"

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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DR estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position, weights=pi_e_at_position, axis=1,
        )
        estimated_rewards += iw * (reward - q_hat_factual)
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

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

        \\hat{V}_{\\mathrm{SNDR}} (\\pi_e; \\mathcal{D}, \\hat{q}) :=
        \\frac{\\mathbb{E}_{\\mathcal{D}}[\\hat{q}(x_t,\\pi_e) +  w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t))]}{\\mathbb{E}_{\\mathcal{D}}[ w(x_t,a_t) ]},

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    Similar to Self-Normalized Inverse Probability Weighting, SNDR estimator applies the self-normalized importance weighting technique to
    increase the stability of the original Doubly Robust estimator.

    Parameters
    ----------
    estimator_name: str, default='sndr'.
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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the SNDR estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position, weights=pi_e_at_position, axis=1,
        )
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        estimated_rewards += iw * (reward - q_hat_factual)
        return estimated_rewards / iw.mean()


@dataclass
class SwitchInverseProbabilityWeighting(DoublyRobust):
    """Estimate the policy value by Switch Inverse Probability Weighting (Switch-IPW).

    Switch-IPW aims to reduce the variance of the Inverse Probability Weighting estimator by using direct method instead
    when the importance weight is large. This estimator estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SwitchIPW}} (\\pi_e; \\mathcal{D}, \\tau)
        := \\mathbb{E}_{\\mathcal{D}} \\left[ \\sum_{a \\in \\mathcal{A}} \\hat{q} (x_t, a) \\pi_e (a|x_t) \\mathbb{I} \\{ w(x_t, a) > \\tau \\}
         + w(x_t,a_t) r_t \\mathbb{I} \\{ w(x_t,a_t) \\le \\tau \\} \\right],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\tau (\\ge 0)` is a switching hyperparameter, which decides the threshold for the importance weight.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    Parameters
    ----------
    tau: float, default=1
        Switching hyperparameter. When importance weight is larger than this parameter, the DM estimator is applied, otherwise the IPW estimator is applied.
        This hyperparameter should be larger than 1., otherwise it is meaningless.

    estimator_name: str, default='switch-ipw'.
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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the Switch-IPW estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        switch_indicator = np.array(iw <= self.tau, dtype=int)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = (1 - switch_indicator) * np.average(
            q_hat_at_position, weights=pi_e_at_position, axis=1,
        )
        estimated_rewards += switch_indicator * iw * reward
        return estimated_rewards


@dataclass
class SwitchDoublyRobust(DoublyRobust):
    """Estimate the policy value by Switch Doubly Robust (Switch-DR).

    Switch-DR aims to reduce the variance of the Doubly Robust estimator by using direct method instead of doubly robust
    when the importance weight is large. This estimator estimates the policy value of a given evaluation policy :math:`\\pi_e` by

    .. math::

        \\hat{V}_{\\mathrm{SwitchDR}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\tau)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w(x_t,a_t) (r_t - \\hat{q}(x_t,a_t)) \\mathbb{I} \\{ w(x_t,a_t) \\le \\tau \\}],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\tau (\\ge 0)` is a switching hyperparameter, which decides the threshold for the importance weight.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    Parameters
    ----------
    tau: float, default=1
        Switching hyperparameter. When importance weight is larger than this parameter, the DM estimator is applied, otherwise the DR estimator is applied.
        This hyperparameter should be larger than 0., otherwise it is meaningless.

    estimator_name: str, default='switch-dr'.
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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the Switch-DR estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        switch_indicator = np.array(iw <= self.tau, dtype=int)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position, weights=pi_e_at_position, axis=1,
        )
        estimated_rewards += switch_indicator * iw * (reward - q_hat_factual)
        return estimated_rewards


@dataclass
class DoublyRobustWithShrinkage(DoublyRobust):
    """Estimate the policy value by Doubly Robust with optimistic shrinkage (DRos).

    DR with shrinkage replaces the importance weight in the original DR estimator with a new weight mapping
    found by directly optimizing sharp bounds on the resulting MSE.

    .. math::

        \\hat{V}_{\\mathrm{DRos}} (\\pi_e; \\mathcal{D}, \\hat{q}, \\lambda)
        := \\mathbb{E}_{\\mathcal{D}} [\\hat{q}(x_t,\\pi_e) +  w_o(x_t,a_t;\\lambda) (r_t - \\hat{q}(x_t,a_t))],

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by :math:`\\pi_b`.
    :math:`w(x,a):=\\pi_e (a|x)/\\pi_b (a|x)` is the importance weight given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_t,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{\\mathcal{D}}[\\cdot]` is the empirical average over :math:`T` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is an estimated expected reward given :math:`x` and :math:`a`.
    To estimate the mean reward function, please use `obp.ope.regression_model.RegressionModel`,
    which supports several fitting methods specific to OPE such as *more robust doubly robust*.

    :math:`w_{o} (x_t,a_t;\\lambda)` is a new weight by the shrinkage technique which is defined as

    .. math::

        w_{o} (x_t,a_t;\\lambda) := \\frac{\\lambda}{w^2(x_t,a_t) + \\lambda} w(x_t,a_t).

    When :math:`\\lambda=0`, we have :math:`w_{o} (x,a;\\lambda)=0` corresponding to the DM estimator.
    In contrast, as :math:`\\lambda \\rightarrow \\infty`, :math:`w_{o} (x,a;\\lambda)` increases and in the limit becomes equal to
    the original importance weight, corresponding to the standard DR estimator.
    Note that there is the other kind of the shrinkage technique called *pessimistic shrinkage*.
    DR with pessimistic shrinkage can be achieved by controlling the clipping hyperparameter of the original DR estimator
    (i.e., obp.ope.DoublyRobust), and thus is not implemented in this class.


    Parameters
    ----------
    lambda_: float
        Shrinkage hyperparameter. This hyperparameter should be larger than 0., otherwise it is meaningless.

    estimator_name: str, default='dr-os'.
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
        reward: array-like, shape (n_rounds,)
            Observed reward (or outcome) of each round, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Distribution over actions or the action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a|x)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated rewards for each round, action, and position by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Rewards estimated by the DRos estimator for each round.

        """
        n_rounds = action.shape[0]
        iw = action_dist[np.arange(n_rounds), action, position] / pscore
        shrinkage_weight = (self.lambda_ * iw) / (iw ** 2 + self.lambda_)
        q_hat_at_position = estimated_rewards_by_reg_model[
            np.arange(n_rounds), :, position
        ]
        q_hat_factual = estimated_rewards_by_reg_model[
            np.arange(n_rounds), action, position
        ]
        pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
        estimated_rewards = np.average(
            q_hat_at_position, weights=pi_e_at_position, axis=1,
        )
        estimated_rewards += shrinkage_weight * (reward - q_hat_factual)
        return estimated_rewards
