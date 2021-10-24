# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Slate Off-Policy Estimators."""
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np

from ..utils import check_iips_inputs
from ..utils import check_rips_inputs
from ..utils import check_sips_inputs
from ..utils import check_cascade_dr_inputs
from ..utils import estimate_confidence_interval_by_bootstrap


@dataclass
class BaseSlateOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for Slate OPE estimators."""

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
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure."""
        raise NotImplementedError


@dataclass
class BaseSlateInverseProbabilityWeighting(BaseSlateOffPolicyEstimator):
    """Base Class of Inverse Probability Weighting Estimators for the slate contextual bandit setting.

    len_list: int (> 1)
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, `len_list=3`.

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
        """Estimate rewards given round (slate_id) and slot (position).

        Parameters
        ----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        behavior_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities of evaluation policy, i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated by IPW given round (slate_id) and slot (position).

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
        """Estimate confidence interval of policy value by nonparametric bootstrap-like procedure.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slates (i.e., rounds or lists of actions).

        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated by IPW given round (slate_id) and slot (position).

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
    """Standard Inverse Propensity Scoring (SIPS) Estimator.

    Note
    -------
    Slate Standard Inverse Propensity Scoring (SIPS) estimates the policy value of evaluation policy :math:`\\pi_e`
    without imposing any assumption on the user behavior.
    Please refer to Eq.(1) in Section 3 of McInerney et al.(2020) for the detail.

    Parameters
    ----------
    estimator_name: str, default='sips'.
        Name of the estimator.

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
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slates (i.e., rounds or lists of actions).

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities of evaluation policy, i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_sips_inputs(
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
            IDs to differentiate slates (i.e., rounds or lists of actions).

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities of evaluation policy, i.e., :math:`\\pi_e(a_t|x_t)`.

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
        check_sips_inputs(
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
    """Independent Inverse Propensity Scoring (IIPS) Estimator.

    Note
    -------
    Slate Independent Inverse Propensity Scoring (IIPS) estimates the policy value of evaluation policy :math:`\\pi_e`
    assuming the item-position click model (rewards observed at each position are assumed to be independent of any other positions.).
    Please refer to Eq.(2) in Section 3 of McInerney et al.(2020) for the detail.

    Parameters
    ----------
    estimator_name: str, default='iips'.
        Name of the estimator.

    References
    ------------
    Shuai Li, Yasin Abbasi-Yadkori, Branislav Kveton, S. Muthukrishnan, Vishwa Vinay, Zheng Wen.
    "Offline Evaluation of Ranking Policies with Click Models", 2018.

    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions", 2020.

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
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slates (i.e., rounds or lists of actions).

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Probabilities that behavior policy selects each action :math:`a` at position (slot) :math:`k` given context :math:`x`, i.e., :math:`\\pi_b(a_{t}(k) |x_t)`.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Probabilities that evaluation policy selects each action :math:`a` at position (slot) :math:`k` given context :math:`x`, i.e., :math:`\\pi_e(a_{t}(k) |x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_iips_inputs(
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
            IDs to differentiate slates (i.e., rounds or lists of actions).

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal action choice probabilities of the slot (:math:`k`) by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_{t, k}|x_t)`.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal action choice probabilities of the slot (:math:`k`) by the evaluation policy, i.e., :math:`\\pi_e(a_{t, k}|x_t)`.

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
        check_iips_inputs(
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
class SlateRewardInteractionIPS(BaseSlateInverseProbabilityWeighting):
    """Reward Interaction Inverse Propensity Scoring (RIPS) Estimator.

    Note
    -------
    Slate Reward Interaction Inverse Propensity Scoring (RIPS) estimates the policy value of evaluation policy :math:`\\pi_e`
    assuming the cascade click model (users interact with actions from the top position to the bottom in a slate).
    Please refer to Eq.(3)-Eq.(4) in Section 3 of McInerney et al.(2020) for the detail.

    Parameters
    ----------
    estimator_name: str, default='rips'.
        Name of the estimator.

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
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slates (i.e., rounds or lists of actions).

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Probabilities that behavior policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_b(a_t(k) | x_t, a_t(1), \ldots, a_t(k-1))`.

        evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Probabilities that evaluation policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_e(a_t(k) | x_t, a_t(1), \ldots, a_t(k-1))`.

        Returns
        ----------
        V_hat: float
            Estimated policy value (performance) of a given evaluation policy.

        """

        check_rips_inputs(
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
            IDs to differentiate slates (i.e., rounds or lists of actions).

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities above the slot (:math:`k`) by a behavior policy (propensity scores), i.e., :math:`\\pi_b(\\{a_{t, j}\\}_{j \\le k}|x_t)`.

        evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities above the slot (:math:`k`) by the evaluation policy, i.e., :math:`\\pi_e(\\{a_{t, j}\\}_{j \\le k}|x_t)`.

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
        check_rips_inputs(
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


@dataclass
class SlateCascadeDoublyRobust(BaseSlateOffPolicyEstimator):
    """Cascade Doubly Robust (Cascade-DR) Estimator.

    Note
    -------
    Slate Cascade Doubly Robust (Cascade-DR) estimates the policy value of evaluation policy :math:`\\pi_e`
    assuming the cascade click model (users interact with actions from the top position to the bottom in a slate).
    It also uses reward prediction :math:`\\hat{Q}_k` as a control variate, which is derived using `obp.ope.SlateRegressionModel`.
    Please refer to Section 3.1 of Kiyohara et al.(2022) for the detail.

    Note that :math:`\\hat{Q}_k` is derived in `obp.ope.SlateRegressionModel`.

    Parameters
    ----------
    len_list: int
        Length of a list of actions recommended in each impression (slate size).
        When Open Bandit Dataset is used, 3 should be set.

    n_unique_action: int
        Number of unique actions.

    estimator_name: str, default='cascade-dr'.
        Name of the estimator.

    References
    ------------
    Haruka Kiyohara, Yuta Saito, Tatsuya Matsuhiro, Yusuke Narita, Nobuyuki Shimizu, and Yasuo Yamamoto.
    "Doubly Robust Off-Policy Evaluation for Ranking Policies under the Cascade Behavior Model.", 2022.

    """

    len_list: int
    n_unique_action: int
    estimator_name: str = "cascade-dr"

    def __post_init__(self):
        """Initialize Class."""
        if self.n_unique_action is None:
            raise ValueError("n_unique_action must be given")

    def _estimate_round_rewards(
        self,
        action: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        q_hat_for_counterfactual_actions: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards given round (slate_id) and slot (position).

        Parameters
        ----------
        action: array-like, (<= n_rounds * len_list,)
            Action observed at each slot in each round of the logged bandit feedback, i.e., :math:`a_{t}(k)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        behavior_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Probabilities that behavior policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_b(a_t(k) | x_t, a_t(1), \\ldots, a_t(k-1))`.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Probabilities that evaluation policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_e(a_t(k) | x_t, a_t(1), \\ldots, a_t(k-1))`.

        q_hat_for_counterfactual_actions: array-like (<= n_rounds * len_list * n_unique_actions, )
            Estimation of :math:`\\hat{Q}_k` for all possible actions
            , i.e., :math:`\\hat{Q}_{t, k}(x_t, a_t(1), \\ldots, a_t(k-1), {a'}_t(k)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

        evaluation_policy_action_dist: array-like (<= n_rounds * len_list * n_unique_actions, )
            Action choice probabilities of evaluation policy for all possible actions
            , i.e., :math:`\\pi_e({a'}_t(k) | x_t, a_t(1), \\ldots, a_t(k-1)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated by Cascade-DR given round (slate_id) and slot (position).

        """
        # (n_rounds_ * len_list * n_unique_action, ) -> (n_rounds_, len_list, n_unique_action)
        q_hat_for_counterfactual_actions_3d = q_hat_for_counterfactual_actions.reshape(
            (-1, self.len_list, self.n_unique_action)
        )
        # \hat{Q} for the action taken by the behavior policy
        # (n_rounds_, len_list, n_unique_action) -> (n_rounds_ * len_list, )
        q_hat_for_taken_action = []
        for i in range(self.n_rounds_):
            for position_ in range(self.len_list):
                q_hat_for_taken_action.append(
                    q_hat_for_counterfactual_actions_3d[
                        i, position_, action[i * self.len_list + position_]
                    ]
                )
        q_hat_for_taken_action = np.array(q_hat_for_taken_action)
        # baseline \hat{Q} by evaluation policy
        # (n_rounds_ * len_list * n_unique_action, ) -> (n_rounds_, len_list, n_unique_action) -> (n_rounds_, len_list) -> (n_rounds_ * len_list, )
        baseline_q_hat_by_eval_policy = (
            (evaluation_policy_action_dist * q_hat_for_counterfactual_actions)
            .reshape((-1, self.len_list, self.n_unique_action))
            .sum(axis=2)
            .flatten()
        )
        # importance weights
        # (n_rounds * len_list, )
        iw = evaluation_policy_pscore / behavior_policy_pscore
        iw_prev = np.roll(iw, 1)
        iw_prev[np.array([i * self.len_list for i in range(self.n_rounds_)])] = 1
        # estimate policy value given each round and slot in a doubly robust manner
        estimated_rewards = (
            iw * (reward - q_hat_for_taken_action)
            + iw_prev * baseline_q_hat_by_eval_policy
        )
        return estimated_rewards

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        q_hat_for_counterfactual_actions: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slates (i.e., rounds or lists of actions).

        action: array-like, (<= n_rounds * len_list,)
            Action observed at each slot in each round of the logged bandit feedback, i.e., :math:`a_{t}(k)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Probabilities that behavior policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_b(a_t(k) | x_t, a_t(1), \\ldots, a_t(k-1))`.

        evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Probabilities that evaluation policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_e(a_t(k) | x_t, a_t(1), \\ldots, a_t(k-1))`.

        q_hat_for_counterfactual_actions: array-like (<= n_rounds * len_list * n_unique_actions, )
            Estimation of :math:`\\hat{Q}_k` for all possible actions
            , i.e., :math:`\\hat{Q}_{t, k}(x_t, a_t(1), \\ldots, a_t(k-1), {a'}_t(k)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

        evaluation_policy_action_dist: array-like (<= n_rounds * len_list * n_unique_actions, )
            Action choice probabilities of evaluation policy for all possible actions
            , i.e., :math:`\\pi_e({a'}_t(k) | x_t, a_t(1), \\ldots, a_t(k-1)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

        Returns
        ----------
        V_hat: array-like, shape (<= n_rounds * len_list,)
            Estimated policy value (performance) of a given evaluation policy.

        """
        check_cascade_dr_inputs(
            n_unique_action=self.n_unique_action,
            slate_id=slate_id,
            action=action,
            reward=reward,
            position=position,
            pscore_cascade=pscore_cascade,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            q_hat_for_counterfactual_actions=q_hat_for_counterfactual_actions,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )
        self.n_rounds_ = np.unique(slate_id).shape[0]
        return (
            self._estimate_round_rewards(
                action=action,
                reward=reward,
                position=position,
                behavior_policy_pscore=pscore_cascade,
                evaluation_policy_pscore=evaluation_policy_pscore_cascade,
                q_hat_for_counterfactual_actions=q_hat_for_counterfactual_actions,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
            ).sum()
            / self.n_rounds_
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        q_hat_for_counterfactual_actions: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slates (i.e., rounds or lists of actions).

        action: array-like, (<= n_rounds * len_list,)
            Action observed at each slot in each round of the logged bandit feedback, i.e., :math:`a_{t}(k)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Probabilities that behavior policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_b(a_t(k) | x_t, a_t(1), \\ldots, a_t(k-1))`.

        evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Probabilities that evaluation policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_e(a_t(k) | x_t, a_t(1), \\ldots, a_t(k-1))`.

        q_hat_for_counterfactual_actions: array-like (<= n_rounds * len_list * n_unique_actions, )
            Estimation of :math:`\\hat{Q}_k` for all possible actions
            , i.e., :math:`\\hat{Q}_{t, k}(x_t, a_t(1), \\ldots, a_t(k-1), {a'}_t(k)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

        evaluation_policy_action_dist: array-like (<= n_rounds * len_list * n_unique_actions, )
            Action choice probabilities of evaluation policy for all possible actions
            , i.e., :math:`\\pi_e({a'}_t(k) | x_t, a_t(1), \\ldots, a_t(k-1)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

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
        check_cascade_dr_inputs(
            slate_id=slate_id,
            action=action,
            reward=reward,
            position=position,
            pscore_cascade=pscore_cascade,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            q_hat_for_counterfactual_actions=q_hat_for_counterfactual_actions,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )
        self.n_rounds_ = np.unique(slate_id).shape[0]
        estimated_rewards = self._estimate_round_rewards(
            action=action,
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_cascade,
            evaluation_policy_pscore=evaluation_policy_pscore_cascade,
            q_hat_for_counterfactual_actions=q_hat_for_counterfactual_actions,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class BaseSlateSelfNormalizedInverseProbabilityWeighting(
    BaseSlateInverseProbabilityWeighting
):
    """Base Class of Self-Normalized Inverse Probability Weighting Estimators for the slate contextual bandit setting.

    len_list: int (> 1)
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, `len_list=3`.

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
        """Self-Normalized estimated rewards given round (slate_id) and slot (position).

        Parameters
        ----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        behavior_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities of evaluation policy, i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Self-Normalized rewards estimated by IPW given round (slate_id) and slot (position).

        """
        estimated_rewards = np.zeros_like(behavior_policy_pscore)
        iw = np.zeros_like(behavior_policy_pscore)
        for position_ in range(self.len_list):
            idx = position == position_
            iw[idx] = evaluation_policy_pscore[idx] / behavior_policy_pscore[idx]
            estimated_rewards[idx] = reward[idx] * iw[idx] / iw[idx].mean()
        return estimated_rewards


@dataclass
class SelfNormalizedSlateStandardIPS(
    SlateStandardIPS, BaseSlateSelfNormalizedInverseProbabilityWeighting
):
    """Self-Normalized Standard Inverse Propensity Scoring (SNSIPS) Estimator.

    Note
    -------
    Self-Normalized Standard Inverse Propensity Scoring (SNSIPS) is our original estimator based on the SlateStandardIPS.

    SNSIPS calculates the empirical average of importance weights
    and re-weights the observed rewards by the empirical average of the importance weights.

    A Self-Normalized estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and increases the stability in some senses.
    See the references for the detailed discussions.

    Parameters
    ----------
    estimator_name: str, default='snsips'.
        Name of the estimator.

    References
    ----------
    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions", 2020.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-normalized Estimator for Counterfactual Learning.", 2015.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "snsips"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards given round (slate_id) and slot (position).

        Parameters
        ----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        position: array-like, shape (<= n_rounds * len_list,)
            IDs to differentiate slot (i.e., position in recommendation/ranking interface) in each slate.

        behavior_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Action choice probabilities of evaluation policy, i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated by the SNSIPS estimator given round (slate_id) and slot (position).

        """
        estimated_rewards = np.zeros_like(behavior_policy_pscore)
        iw = evaluation_policy_pscore / behavior_policy_pscore
        estimated_rewards = reward * iw / iw.mean()
        return estimated_rewards


@dataclass
class SelfNormalizedSlateIndependentIPS(
    SlateIndependentIPS, BaseSlateSelfNormalizedInverseProbabilityWeighting
):
    """Self-Normalized Independent Inverse Propensity Scoring (SNIIPS) Estimator.

    Note
    -------
    Self-Normalized Independent Inverse Propensity Scoring (SNIIPS) is our original estimator based on the SlateIndependentIPS.

    SNIIPS calculates the slot-level empirical average of importance weights
    and re-weights the observed rewards of slot :math:`k` by the empirical average of the importance weights.

    A Self-Normalized estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and increases the stability in some senses.
    See the references for the detailed discussions.

    Parameters
    ----------
    estimator_name: str, default='sniips'.
        Name of the estimator.

    References
    ----------
    Shuai Li, Yasin Abbasi-Yadkori, Branislav Kveton, S. Muthukrishnan, Vishwa Vinay, Zheng Wen.
    "Offline Evaluation of Ranking Policies with Click Models", 2018.

    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions", 2020.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-normalized Estimator for Counterfactual Learning.", 2015.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "sniips"


@dataclass
class SelfNormalizedSlateRewardInteractionIPS(
    SlateRewardInteractionIPS, BaseSlateSelfNormalizedInverseProbabilityWeighting
):
    """Self-Normalized Reward Interaction Inverse Propensity Scoring (SNRIPS) Estimator.

    Note
    -------
    Self-Normalized Reward Interaction Inverse Propensity Scoring (SNRIPS) is the self-normalized version of SlateRewardInteractionIPS.

    SNRIPS calculates the slot-level empirical average of importance weights
    and re-weights the observed rewards of slot :math:`k` by the empirical average of the importance weights.

    A Self-Normalized estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and increases the stability in some senses.
    See the references for the detailed discussions.

    Parameters
    ----------
    estimator_name: str, default='snrips'.
        Name of the estimator.

    References
    ----------
    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions", 2020.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-normalized Estimator for Counterfactual Learning.", 2015.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "snrips"
