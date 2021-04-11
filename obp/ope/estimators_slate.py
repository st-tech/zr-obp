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
    """Base class for OPE estimators."""

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
    """Base Class of Slate Inverse Probability Weighting."""

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        position_weight: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each round.

        Parameters
        ----------
        reward: array-like or Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        pscore: array-like or Tensor, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like or Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like or Tensor, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.

        Returns
        ----------
        estimated_rewards: array-like or Tensor, shape (n_rounds,)
            Rewards estimated by IPW for each round.

        """
        reward_weight = np.vectorize(lambda x: position_weight[x])(position)
        iw = evaluation_policy_pscore / behavior_policy_pscore
        return reward * iw * reward_weight

    @staticmethod
    def _extract_reward_by_bootstrap(
        slate_id: np.ndarray, reward: np.ndarray, sampled_slate: np.ndarray
    ) -> np.ndarray:
        sampled_reward = list()
        for slate in sampled_slate:
            sampled_reward.extend(reward[slate_id == slate])
        return np.array(sampled_reward)

    def _estimate_slate_confidence_interval_by_bootstrap(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
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
            sampled_reward = self._extract_reward_by_bootstrap(
                slate_id=slate_id, reward=reward, sampled_slate=sampled_slate
            )
            boot_samples.append(sampled_reward.sum() / unique_slate.shape[0])
        lower_bound = np.percentile(boot_samples, 100 * (alpha / 2))
        upper_bound = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))
        return {
            "mean": np.mean(boot_samples),
            f"{100 * (1. - alpha)}% CI (lower)": lower_bound,
            f"{100 * (1. - alpha)}% CI (upper)": upper_bound,
        }


@dataclass
class SlateStandardIPS(BaseSlateInverseProbabilityWeighting):
    """Estimate the policy value by Slate Standard Inverse Probability Scoring (SIPS)."""

    len_list: int
    estimator_name: str = "sips"
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

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> float:
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
        reward = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
            position_weight=self.position_weight,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id, reward=reward
        )


@dataclass
class SlateIndependentIPS(BaseSlateInverseProbabilityWeighting):
    """Estimate the policy value by Slate Independent Inverse Probability Scoring (IIPS)."""

    len_list: int
    estimator_name: str = "iips"
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
        reward = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_item_position,
            evaluation_policy_pscore=evaluation_policy_pscore,
            position_weight=self.position_weight,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id, reward=reward
        )


@dataclass
class SlateRecursiveIPS(BaseSlateInverseProbabilityWeighting):
    """Estimate the policy value by Slate Recursive Inverse Probability Scoring (RIPS)."""

    len_list: int
    estimator_name: str = "rips"
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
        reward = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_cascade,
            evaluation_policy_pscore=evaluation_policy_pscore,
            position_weight=self.position_weight,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id, reward=reward
        )
