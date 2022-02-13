# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators for Slate/Ranking Policies."""
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar

from ..utils import check_cascade_dr_inputs
from ..utils import check_iips_inputs
from ..utils import check_rips_inputs
from ..utils import check_sips_inputs
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
        """Estimate the confidence interval of the policy value using bootstrap."""
        raise NotImplementedError


@dataclass
class BaseSlateInverseProbabilityWeighting(BaseSlateOffPolicyEstimator):
    """Base Class of Inverse Probability Weighting Estimators for the slate contextual bandit setting.

    len_list: int (> 1)
        Length of a list of actions in a recommendation/ranking inferface, slate size.
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
        """Estimate rewards for each slate and slot (position).

        Parameters
        ----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        behavior_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of behavior policy choosing a particular action at each position (slot) or
            joint probabilities of behavior policy choosing a whole slate/ranking of actions.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each slot/position or
            joint probabilities of evaluation policy choosing a whole slate/ranking of actions.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated for each slate and slot (position).

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
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated for each slate and slot (position).

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
    SIPS estimates the policy value of evaluation policy :math:`\\pi_e`
    without imposing any assumption on user behavior.
    Please refer to Eq.(1) in Section 3 of McInerney et al.(2020) for more details.

    Parameters
    ----------
    estimator_name: str, default='sips'.
        Name of the estimator.

    References
    ------------
    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions.", 2020.

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
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a slate action, i.e., :math:`\\pi_b(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

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
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a slate action, i.e., :math:`\\pi_b(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

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
    IIPS estimates the policy value of evaluation policy :math:`\\pi_e` assuming
    the item-position click model (rewards observed at each position are assumed to be independent of any other positions.).
    Please refer to Eq.(2) in Section 3 of McInerney et al.(2020) for more details.

    Parameters
    ----------
    estimator_name: str, default='iips'.
        Name of the estimator.

    References
    ------------
    Shuai Li, Yasin Abbasi-Yadkori, Branislav Kveton, S. Muthukrishnan, Vishwa Vinay, Zheng Wen.
    "Offline Evaluation of Ranking Policies with Click Models", 2018.

    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions.", 2020.

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
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of behavior policy choosing a particular action at each position (slot),
            i.e., :math:`\\pi_b(a_{i}(l) |x_i)`.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each position (slot),
             i.e., :math:`\\pi_e(a_{i}(l) |x_i)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

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
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of behavior policy choosing a particular action at each position (slot),
            i.e., :math:`\\pi_b(a_{i}(l) |x_i)`.

        evaluation_policy_pscore_item_position: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each position (slot),
             i.e., :math:`\\pi_e(a_{i}(l) |x_i)`.

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
    RIPS estimates the policy value of evaluation policy :math:`\\pi_e` assuming
    the cascade click model (users interact with actions from the top position to the bottom in a slate).
    Please refer to Eq.(3)-Eq.(4) in Section 3 of McInerney et al.(2020) for more details.

    Parameters
    ----------
    estimator_name: str, default='rips'.
        Name of the estimator.

    References
    ------------
    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions.", 2020.

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
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

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
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_pscore_cascade: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

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
    Cascade-DR estimates the policy value of evaluation (ranking) policy :math:`\\pi_e` assuming
    the cascade click model (users interact with actions from the top position to the bottom in a slate).
    It also uses the Q function estimates :math:`\\hat{Q}_l` as a control variate, which is derived using `obp.ope.SlateRegressionModel`.
    Please refer to Section 3.1 of Kiyohara et al.(2022) for more details.

    Parameters
    ----------
    len_list: int
        Length of a list of actions in a recommendation/ranking interface, slate size.
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
        check_scalar(self.n_unique_action, "n_unique_action", int, min_val=1)

    def _estimate_round_rewards(
        self,
        action: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        q_hat: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each slate and slot (position).

        Parameters
        ----------
        action: array-like, (n_rounds * len_list,)
            Actions observed at each slot in a ranking/slate in logged bandit data, i.e., :math:`a_{i}(l)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds * len_list,)
            Slot-level rewards observed for each data in logged bandit data, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        q_hat: array-like (n_rounds * len_list * n_unique_actions, )
            :math:`\\hat{Q}_l` for all unique actions,
            i.e., :math:`\\hat{Q}_{i,l}(x_i, a_i(1), \\ldots, a_i(l-1), a_i(l)) \\forall a_i(l) \\in \\mathcal{A}`.

        evaluation_policy_action_dist: array-like (n_rounds * len_list * n_unique_actions, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds * len_list,)
            Rewards rewards for each slate and slot (position).

        """
        # (n_rounds_ * len_list * n_unique_action, ) -> (n_rounds_, len_list, n_unique_action)
        q_hat_3d = q_hat.reshape((-1, self.len_list, self.n_unique_action))
        # the estimated Q functions for the action taken by the behavior policy
        # (n_rounds_, len_list, n_unique_action) -> (n_rounds_ * len_list, )
        q_hat_for_observed_action = []
        for i in range(self.n_rounds_):
            for pos_ in range(self.len_list):
                q_hat_for_observed_action.append(
                    q_hat_3d[i, pos_, action[i * self.len_list + pos_]]
                )
        q_hat_for_observed_action = np.array(q_hat_for_observed_action)
        # the expected Q function under the evaluation policy
        # (n_rounds_ * len_list * n_unique_action, ) -> (n_rounds_, len_list, n_unique_action) -> (n_rounds_, len_list) -> (n_rounds_ * len_list, )
        expected_q_hat_under_eval_policy = (
            (evaluation_policy_action_dist * q_hat)
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
            iw * (reward - q_hat_for_observed_action)
            + iw_prev * expected_q_hat_under_eval_policy
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
        q_hat: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        action: array-like, (n_rounds * len_list,)
            Actions observed at each slot in a ranking/slate in logged bandit data, i.e., :math:`a_{i}(l)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds * len_list,)
            Slot-level rewards observed for each data in logged bandit data, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`).

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.


        q_hat: array-like (n_rounds * len_list * n_unique_actions, )
            :math:`\\hat{Q}_l` for all unique actions,
            i.e., :math:`\\hat{Q}_{i,l}(x_i, a_i(1), \\ldots, a_i(l-1), a_i(l)) \\forall a_i(l) \\in \\mathcal{A}`.

        evaluation_policy_action_dist: array-like (n_rounds * len_list * n_unique_actions, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

        Returns
        ----------
        V_hat: array-like, shape (n_rounds * len_list,)
            Estimated policy value of evaluation policy.

        """
        check_cascade_dr_inputs(
            n_unique_action=self.n_unique_action,
            slate_id=slate_id,
            action=action,
            reward=reward,
            position=position,
            pscore_cascade=pscore_cascade,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            q_hat=q_hat,
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
                q_hat=q_hat,
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
        q_hat: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        action: array-like, (n_rounds * len_list,)
            Actions observed at each slot in a ranking/slate in logged bandit data, i.e., :math:`a_{i}(l)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds * len_list,)
            Slot-level rewards observed for each data in logged bandit data, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`).

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.


        q_hat: array-like (n_rounds * len_list * n_unique_actions, )
            :math:`\\hat{Q}_l` for all unique actions,
            i.e., :math:`\\hat{Q}_{i,l}(x_i, a_i(1), \\ldots, a_i(l-1), a_i(l)) \\forall a_i(l) \\in \\mathcal{A}`.

        evaluation_policy_action_dist: array-like (n_rounds * len_list * n_unique_actions, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

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
        check_cascade_dr_inputs(
            n_unique_action=self.n_unique_action,
            slate_id=slate_id,
            action=action,
            reward=reward,
            position=position,
            pscore_cascade=pscore_cascade,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            q_hat=q_hat,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )
        self.n_rounds_ = np.unique(slate_id).shape[0]
        estimated_rewards = self._estimate_round_rewards(
            action=action,
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore_cascade,
            evaluation_policy_pscore=evaluation_policy_pscore_cascade,
            q_hat=q_hat,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )

    def _estimate_slate_confidence_interval_by_bootstrap(
        self,
        slate_id: np.ndarray,
        estimated_rewards: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated for each slate and slot (position).

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
class BaseSlateSelfNormalizedInverseProbabilityWeighting(
    BaseSlateInverseProbabilityWeighting
):
    """Base Class of Self-Normalized Inverse Probability Weighting Estimators for the slate contextual bandit setting.

    len_list: int (> 1)
        Length of a list of actions in a recommendation/ranking inferface, slate size.
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
        """Self-Normalized estimated rewards for each slate and slot (position).

        Parameters
        ----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        behavior_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of behavior policy choosing a particular action at each position (slot) or
            joint probabilities of behavior policy choosing a whole slate/ranking of actions.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each slot/position or
            joint probabilities of evaluation policy choosing a whole slate/ranking of actions.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Self-Normalized rewards estimated for each slate and slot (position).

        """
        estimated_rewards = np.zeros_like(behavior_policy_pscore)
        iw = np.zeros_like(behavior_policy_pscore)
        for pos_ in range(self.len_list):
            idx = position == pos_
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
    SNSIPS is the self-normalized version of `obp.ope.SlateStandardIPS`.

    SNSIPS calculates the empirical average of importance weights
    and normalizes the observed slate-level rewards by the empirical average of the importance weights.

    A Self-Normalized estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and gains some stability in OPE.
    See the reference papers for more details.

    Parameters
    ----------
    estimator_name: str, default='snsips'.
        Name of the estimator.

    References
    ----------
    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions.", 2020.

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
        """Estimate rewards for each slate and slot (position).

        Parameters
        ----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a slate action, i.e., :math:`\\pi_b(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated by the SNSIPS estimator for each slate and slot (position).

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
    SNIIPS is the self-normalized version of `obp.ope.SlateIndependentIPS`.

    SNIIPS calculates the slot-level empirical average of importance weights
    and normalizes the observed slot-level rewards by the empirical average of the importance weights.

    A Self-Normalized estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and gains some stability in OPE.
    See the reference papers for more details.

    Parameters
    ----------
    estimator_name: str, default='sniips'.
        Name of the estimator.

    References
    ----------
    Shuai Li, Yasin Abbasi-Yadkori, Branislav Kveton, S. Muthukrishnan, Vishwa Vinay, Zheng Wen.
    "Offline Evaluation of Ranking Policies with Click Models", 2018.

    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions.", 2020.

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
    SNRIPS is the self-normalized version of `obp.ope.SlateRewardInteractionIPS`.

    SNRIPS calculates the slot-level empirical average of the importance weights
    and normalizes the observed slot-level rewards by the empirical average of the importance weights.

    A Self-Normalized estimator is not unbiased even when the behavior policy is known.
    However, it is still consistent for the true policy value and gains some stability in OPE.
    See the reference papers for more details.

    Parameters
    ----------
    estimator_name: str, default='snrips'.
        Name of the estimator.

    References
    ----------
    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions.", 2020.

    Adith Swaminathan and Thorsten Joachims.
    "The Self-normalized Estimator for Counterfactual Learning.", 2015.

    Nathan Kallus and Masatoshi Uehara.
    "Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning.", 2019.

    """

    estimator_name: str = "snrips"
