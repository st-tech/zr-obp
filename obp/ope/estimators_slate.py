# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Slate Off-Policy Estimators."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..utils import (
    estimate_confidence_interval_by_bootstrap,
    check_sips_ope_inputs,
    check_iips_ope_inputs,
    check_rips_ope_inputs,
)


@dataclass
class BaseSlateOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for Slate OPE estimators."""

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
class BaseSlateInverseProbabilityWeighting(BaseSlateOffPolicyEstimator):
    """Base Class of Slate Inverse Probability Weighting.

    len_list: int (> 1)
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    """

    len_list: int

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round and slot.

        Parameters
        ----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed in each round and slot of the logged bandit feedback, i.e., :math:`r_{t, k}`.

        position: array-like, shape (<= n_rounds * len_list,)
            Positions of each round and slot in the given logged bandit feedback.

        behavior_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities by the evaluation policy (propensity scores), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated by IPW for each round and slot.

        """
        iw = evaluation_policy_pscore / behavior_policy_pscore
        estimated_rewards = reward * iw
        return estimated_rewards

    def _estimate_slate_confidence_interval_by_bootstrap(
        self,
        slate_id: np.ndarray,
        estimated_rewards: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """Estimate confidence interval by nonparametric bootstrap-like procedure.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate id observed in each round of the logged bandit feedback.

        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated by IPW for each round and slot.

        alpha: float, default=0.05
            Significant level of confidence intervals.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        unique_slate = np.unique(slate_id)
        # sum estimated_rewards in each slate
        estimated_round_rewards = list()
        for slate in unique_slate:
            estimated_round_rewards.append(estimated_rewards[slate_id == slate].sum())
        estimated_round_rewards = np.array(estimated_round_rewards)
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateStandardIPS(BaseSlateInverseProbabilityWeighting):
    """Estimate the policy value by Slate Standard Inverse Propensity Scoring (SIPS).

    Note
    -------
    Slate Standard Inverse Propensity Scoring (SIPS) estimates the policy value of a given evaluation policy :math:`\\pi_e`.

    Parameters
    ----------
    estimator_name: str, default='sips'.
        Name of off-policy estimator.

    References
    ------------
    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions", 2020.

    """

    estimator_name: str = "sips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate id observed in each round of the logged bandit feedback.

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed in each round and slot of the logged bandit feedback, i.e., :math:`r_{t, k}`.

        position: array-like, shape (<= n_rounds * len_list,)
            Positions of each round and slot in the given logged bandit feedback.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities by the evaluation policy (propensity scores), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_sips_ope_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        return (
            self._estimate_round_rewards(
                reward=reward,
                position=position,
                behavior_policy_pscore=pscore,
                evaluation_policy_pscore=evaluation_policy_pscore,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate id observed in each round of the logged bandit feedback.

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed in each round and slot of the logged bandit feedback, i.e., :math:`r_{t, k}`.

        position: array-like, shape (<= n_rounds * len_list,)
            Positions of each round and slot in the given logged bandit feedback.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities by the evaluation policy (propensity scores), i.e., :math:`\\pi_e(a_t|x_t)`.

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
        check_sips_ope_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateIndependentIPS(BaseSlateInverseProbabilityWeighting):
    """Estimate the policy value by Slate Independent Inverse Propensity Scoring (IIPS).

    Note
    -------
    Slate Independent Inverse Propensity Scoring (IIPS) estimates the policy value of a given evaluation policy :math:`\\pi_e`.

    Parameters
    ----------
    estimator_name: str, default='iips'.
        Name of off-policy estimator.

    References
    ------------
    Shuai Li, Yasin Abbasi-Yadkori, Branislav Kveton, S. Muthukrishnan, Vishwa Vinay, Zheng Wen.
    "Offline Evaluation of Ranking Policies with Click Models", 2018.

    """

    estimator_name: str = "iips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate id observed in each round of the logged bandit feedback.

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed in each round and slot of the logged bandit feedback, i.e., :math:`r_{t, k}`.

        position: array-like, shape (<= n_rounds * len_list,)
            Positions of each round and slot in the given logged bandit feedback.

        pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal action choice probabilities of the slot (:math:`k`) by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_{t, k}|x_t)`.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal action choice probabilities of the slot (:math:`k`) by the evaluation policy (propensity scores), i.e., :math:`\\pi_e(a_{t, k}|x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_iips_ope_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_item_position=pscore_item_position,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
        )
        return (
            self._estimate_round_rewards(
                reward=reward,
                position=position,
                behavior_policy_pscore=pscore_item_position,
                evaluation_policy_pscore=evaluation_policy_pscore_item_position,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate id observed in each round of the logged bandit feedback.

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed in each round and slot of the logged bandit feedback, i.e., :math:`r_{t, k}`.

        position: array-like, shape (<= n_rounds * len_list,)
            Positions of each round and slot in the given logged bandit feedback.

        pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal action choice probabilities of the slot (:math:`k`) by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_{t, k}|x_t)`.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal action choice probabilities of the slot (:math:`k`) by the evaluation policy (propensity scores), i.e., :math:`\\pi_e(a_{t, k}|x_t)`.

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
        check_iips_ope_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_item_position=pscore_item_position,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_item_position,
            evaluation_policy_pscore=evaluation_policy_pscore_item_position,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SlateRecursiveIPS(BaseSlateInverseProbabilityWeighting):
    """Estimate the policy value by Slate Recursive Inverse Propensity Scoring (RIPS).

    Note
    -------
    Slate Recursive Inverse Propensity Scoring (RIPS) estimates the policy value of a given evaluation policy :math:`\\pi_e`.

    Parameters
    ----------
    estimator_name: str, default='rips'.
        Name of off-policy estimator.

    References
    ------------
    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions", 2020.

    """

    estimator_name: str = "rips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate id observed in each round of the logged bandit feedback.

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed in each round and slot of the logged bandit feedback, i.e., :math:`r_{t, k}`.

        position: array-like, shape (<= n_rounds * len_list,)
            Positions of each round and slot in the given logged bandit feedback.

        pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities above the slot (:math:`k`) by a behavior policy (propensity scores), i.e., :math:`\\pi_b(\\{a_{t, j}\\}_{j \\le k}|x_t)`.

        evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities above the slot (:math:`k`) by the evaluation policy (propensity scores), i.e., :math:`\\pi_e(\\{a_{t, j}\\}_{j \\le k}|x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """

        check_rips_ope_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_cascade=pscore_cascade,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
        )
        return (
            self._estimate_round_rewards(
                reward=reward,
                position=position,
                behavior_policy_pscore=pscore_cascade,
                evaluation_policy_pscore=evaluation_policy_pscore_cascade,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate id observed in each round of the logged bandit feedback.

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed in each round and slot of the logged bandit feedback, i.e., :math:`r_{t, k}`.

        position: array-like, shape (<= n_rounds * len_list,)
            Positions of each round and slot in the given logged bandit feedback.

        pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities above the slot (:math:`k`) by a behavior policy (propensity scores), i.e., :math:`\\pi_b(\\{a_{t, j}\\}_{j \\le k}|x_t)`.

        evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities above the slot (:math:`k`) by the evaluation policy (propensity scores), i.e., :math:`\\pi_e(\\{a_{t, j}\\}_{j \\le k}|x_t)`.

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
        check_rips_ope_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore_cascade=pscore_cascade,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_cascade,
            evaluation_policy_pscore=evaluation_policy_pscore_cascade,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
