# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Slate Off-Policy Estimators."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.utils import check_random_state

from ..utils import check_confidence_interval_arguments


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


# TODO :comment
@dataclass
class BaseSlateInverseProbabilityWeighting(BaseSlateOffPolicyEstimator):
    """Base Class of Slate Inverse Probability Weighting.

    len_list: int (> 1)
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    position_weight: array-like, shape (len_list,)
        Non-negative weight for each slot.

    """

    len_list: int
    position_weight: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.position_weight is None:
            self.position_weight = np.ones(self.len_list)
        else:
            if not isinstance(self.position_weight, np.ndarray):
                raise ValueError("position weight type")
            if not (
                self.position_weight.ndim == 1
                and self.position_weight.shape[0] == self.len_list
            ):
                raise ValueError("position weight shape")

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        position_weight: np.ndarray,
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

        position_weight: array-like, shape (len_list,)
            Non-negative weight for each slot.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated by IPW for each round and slot (weighted based on position_weight).

        """
        reward_weight = np.vectorize(lambda x: position_weight[x])(position)
        iw = evaluation_policy_pscore / behavior_policy_pscore
        estimated_rewards = reward * iw * reward_weight
        return estimated_rewards

    @staticmethod
    def _extract_reward_by_bootstrap(
        slate_id: np.ndarray, estimated_rewards: np.ndarray, sampled_slate: np.ndarray
    ) -> np.ndarray:
        """Extract reward based on sampled slate.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate id observed in each round of the logged bandit feedback.

        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated by IPW for each round and slot (weighted based on position_weight).

        sampled_slate: array-like, shape (n_rounds,)
            Slate id sampled by bootstrap.

        Returns
        ----------
        sampled_estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Estimated rewards sampled by bootstrap

        """
        sampled_estimated_rewards = list()
        for slate in sampled_slate:
            sampled_estimated_rewards.extend(estimated_rewards[slate_id == slate])
        sampled_estimated_rewards = np.array(sampled_estimated_rewards)
        return sampled_estimated_rewards

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
            Rewards estimated by IPW for each round and slot (weighted based on position_weight).

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
        check_confidence_interval_arguments(
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

        unique_slate = np.unique(slate_id)
        boot_samples = list()
        random_ = check_random_state(random_state)
        for _ in np.arange(n_bootstrap_samples):
            sampled_slate = random_.choice(unique_slate, size=unique_slate.shape[0])
            sampled_estimated_rewards = self._extract_reward_by_bootstrap(
                slate_id=slate_id,
                estimated_rewards=estimated_rewards,
                sampled_slate=sampled_slate,
            )
            boot_samples.append(sampled_estimated_rewards.sum() / unique_slate.shape[0])
        lower_bound = np.percentile(boot_samples, 100 * (alpha / 2))
        upper_bound = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))
        return {
            "mean": np.mean(boot_samples),
            f"{100 * (1. - alpha)}% CI (lower)": lower_bound,
            f"{100 * (1. - alpha)}% CI (upper)": upper_bound,
        }


@dataclass
class SlateStandardIPS(BaseSlateInverseProbabilityWeighting):
    """Estimate the policy value by Slate Standard Inverse Probability Scoring (SIPS).

    Note
    -------
    Slate Standard Inverse Probability Scoring (SIPS) estimates the policy value of a given evaluation policy :math:`\\pi_e` by

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
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed in each round and slot of the logged bandit feedback, i.e., :math:`r_{t, k}`.

        position: array-like, shape (<= n_rounds * len_list,)
            Positions of each round and slot in the given logged bandit feedback.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities by the evaluation policy (propensity scores), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        return self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
            position_weight=self.position_weight,
        ).mean()

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
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
            position_weight=self.position_weight,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id, estimated_rewards=estimated_rewards
        )


@dataclass
class SlateIndependentIPS(BaseSlateInverseProbabilityWeighting):
    """Estimate the policy value by Slate Independent Inverse Probability Scoring (IIPS)."""

    estimator_name: str = "iips"

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> float:
        return self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_item_position,
            evaluation_policy_pscore=evaluation_policy_pscore,
            position_weight=self.position_weight,
        ).mean()

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_item_position,
            evaluation_policy_pscore=evaluation_policy_pscore,
            position_weight=self.position_weight,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id, estimated_rewards=estimated_rewards
        )


@dataclass
class SlateRecursiveIPS(BaseSlateInverseProbabilityWeighting):
    """Estimate the policy value by Slate Recursive Inverse Probability Scoring (RIPS)."""

    estimator_name: str = "rips"

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> float:
        return self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_cascade,
            evaluation_policy_pscore=evaluation_policy_pscore,
            position_weight=self.position_weight,
        ).mean()

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_cascade,
            evaluation_policy_pscore=evaluation_policy_pscore,
            position_weight=self.position_weight,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id, estimated_rewards=estimated_rewards
        )
