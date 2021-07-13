# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic Slate Logged Bandit Feedback."""
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Union
from itertools import permutations, product

import numpy as np
from scipy.stats import truncnorm
from scipy.special import perm
from sklearn.utils import check_random_state, check_scalar
from tqdm import tqdm

from .base import BaseBanditDataset
from ..types import BanditFeedback
from ..utils import softmax, sigmoid


@dataclass
class SyntheticSlateBanditDataset(BaseBanditDataset):
    """Class for generating synthetic slate bandit dataset.

    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times,
    we have different bandit samples with the same setting.
    This can be used to estimate confidence intervals of the performances of Slate OPE estimators.

    If None is set as `behavior_policy_function`, the synthetic data will be context-free bandit feedback.

    Parameters
    -----------
    n_unique_action: int (>= len_list)
        Number of unique actions.

    len_list: int (> 1)
        Length of a list of actions recommended in each slate.
        When Open Bandit Dataset is used, 3 should be set.

    dim_context: int, default=1
        Number of dimensions of context vectors.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary' is given, rewards are sampled from the Bernoulli distribution.
        When 'continuous' is given, rewards are sampled from the truncated Normal distribution with `scale=1`.
        The mean parameter of the reward distribution is determined by the `reward_function` specified by the next argument.

    reward_structure: str, default='cascade_additive'
        Type of reward structure, which must be one of 'cascade_additive', 'cascade_decay', 'independent', 'standard_additive', or 'standard_decay'.
        When 'cascade_additive' or 'standard_additive' is given, additive action_interaction_weight_matrix (:math:`W \\in \\mathbb{R}^{\\text{n_unique_action} \\times \\text{n_unique_action}}`) is generated.
        When 'cascade_decay', 'standard_decay', or 'independent' is given, decay action_interaction_weight_matrix (:math:`\\in \\mathbb{R}^{\\text{len_list} \\times \\text{len_list}}`) is generated.
        Expected reward is calculated as follows (:math:`f` is a base reward function of each item-position, :math:`g` is a transform function, and :math:`h` is a decay function):
            'cascade_additive': :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j < k} W(a(k), a(j)))`.
            'cascade_decay': :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) - \\sum_{j < k} g^{-1}(f(x, a(j))) / h(|k-j|))`.
            'independent': :math:`q_k(x, a) = f(x, a(k))`
            'standard_additive': :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j \\neq k} W(a(k), a(j)))`.
            'standard_decay': :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) - \\sum_{j \\neq k} g^{-1}(f(x, a(j))) / h(|k-j|))`.
        When reward_type is 'continuous', transform function is the identity function.
        When reward_type is 'binary', transform function is the logit function.

    decay_function: str, default='exponential'
        Type of decay function, which must be one of 'exponential' or 'inverse'.
        Decay function used for 'cascade_decay' and 'standard_decay' reward structures.
        Discount rate is calculated as follows (:math:`k` and :math:`j` are positions of the two slots).
            'exponential': :math:`h(|k-j|) = \\exp(-|k-j|)`.
            'inverse': :math:`h(|k-j|) = \\frac{1}{|k-j|+1})`.

    click_model: str, default=None
        Type of click model, which must be one of None, 'pbm', or 'cascade'.
        When None is given, reward at each slot is sampled based on the original expected rewards.
        When 'pbm' is given, reward at each slot is sampled based on the position-based model.
        When 'cascade' is given, reward at each slot is sampled based on the cascade model.
        When using some click model, 'continuous' reward type is unavailable.

    eta: float, default=1.0
        A hyperparameter to define the click models.
        When click_model='pbm', then eta defines the examination probabilities of the position-based model.
        For example, when eta=0.5, then the examination probability at position `k` is :math:`\\theta (k) = (1/k)^{0.5}`.
        When click_model='cascade', then eta defines the position-dependent attractiveness parameters of the dependent click model
        (an extension of the cascade model).
        For example, when eta=0.5, the position-dependent attractiveness parameter at position `k` is :math:`\\alpha (k) = (1/k)^{0.5}`.
        When eta is very large, the click model induced by eta is close to the original cascade model.

    base_reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default=None
        Function generating expected reward for each given action-context pair,
        i.e., :math:`\\mu: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None is set, context **independent** expected reward for each action will be
        sampled from the uniform distribution automatically.

    behavior_policy_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default=None
        Function generating logit value of each action in action space,
        i.e., :math:`\\f: \\mathcal{X} \\rightarrow \\mathbb{R}^{\\mathcal{A}}`.
        If None is set, context **independent** uniform distribution will be used (uniform behavior policy).

    is_factorizable: bool
        A boolean parameter whether to use factorizable evaluation policy (which choose slot actions independently) or not.
        When `n_unique_action` and `len_list` are large, this parameter should be set to True because of the computational time.

    random_state: int, default=12345
        Controls the random seed in sampling synthetic slate bandit dataset.

    dataset_name: str, default='synthetic_slate_bandit_dataset'
        Name of the dataset.

    ----------

    .. code-block:: python

        >>> from obp.dataset import (
            logistic_reward_function,
            linear_behavior_policy_logit,
            SyntheticSlateBanditDataset,
        )

        # generate synthetic contextual bandit feedback with 10 actions.
        >>> dataset = SyntheticSlateBanditDataset(
                n_unique_action=10,
                dim_context=5,
                len_list=3,
                base_reward_function=logistic_reward_function,
                behavior_policy_function=linear_behavior_policy_logit,
                reward_type='binary',
                reward_structure='cascade_additive',
                click_model='cascade',
                random_state=12345
            )
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(
                n_rounds=5, return_pscore_item_position=True
            )
        >>> bandit_feedback
        {
            'n_rounds': 5,
            'n_unique_action': 10,
            'slate_id': array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
            'context': array([[-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057],
                [ 1.39340583,  0.09290788,  0.28174615,  0.76902257,  1.24643474],
                [ 1.00718936, -1.29622111,  0.27499163,  0.22891288,  1.35291684],
                [ 0.88642934, -2.00163731, -0.37184254,  1.66902531, -0.43856974],
                [-0.53974145,  0.47698501,  3.24894392, -1.02122752, -0.5770873 ]]),
            'action_context': array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]),
            'action': array([8, 6, 5, 4, 7, 0, 1, 3, 5, 4, 6, 1, 4, 1, 7]),
            'position': array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
            'reward': array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]),
            'expected_reward_factual': array([0.5       , 0.73105858, 0.5       , 0.88079708, 0.88079708,
                0.88079708, 0.5       , 0.73105858, 0.5       , 0.5       ,
                0.26894142, 0.5       , 0.73105858, 0.73105858, 0.5       ]),
            'pscore_cascade': array([0.05982646, 0.00895036, 0.00127176, 0.10339675, 0.00625482,
                0.00072447, 0.14110696, 0.01868618, 0.00284884, 0.10339675,
                0.01622041, 0.00302774, 0.10339675, 0.01627253, 0.00116824]),
            'pscore': array([0.00127176, 0.00127176, 0.00127176, 0.00072447, 0.00072447,
                0.00072447, 0.00284884, 0.00284884, 0.00284884, 0.00302774,
                0.00302774, 0.00302774, 0.00116824, 0.00116824, 0.00116824]),
            'pscore_item_position': array([0.19068462, 0.40385939, 0.33855573, 0.31231088, 0.40385939,
                0.2969341 , 0.40489767, 0.31220474, 0.3388982 , 0.31231088,
                0.33855573, 0.40489767, 0.31231088, 0.40489767, 0.33855573])
        }

    """

    n_unique_action: int
    len_list: int
    dim_context: int = 1
    reward_type: str = "binary"
    reward_structure: str = "cascade_additive"
    decay_function: str = "exponential"
    click_model: Optional[str] = None
    eta: float = 1.0
    base_reward_function: Optional[
        Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ]
    ] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    is_factorizable: bool = False
    random_state: int = 12345
    dataset_name: str = "synthetic_slate_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        if not isinstance(self.n_unique_action, int) or self.n_unique_action <= 1:
            raise ValueError(
                f"n_unique_action must be an integer larger than 1, but {self.n_unique_action} is given"
            )
        if not isinstance(self.len_list, int) or self.len_list <= 1:
            raise ValueError(
                f"len_list must be an integer larger than 1, but {self.len_list} is given"
            )
        if not self.is_factorizable and self.len_list > self.n_unique_action:
            raise ValueError(
                f"len_list must be equal to or smaller than n_unique_action, but {self.len_list} is given"
            )
        if not isinstance(self.dim_context, int) or self.dim_context <= 0:
            raise ValueError(
                f"dim_context must be a positive integer, but {self.dim_context} is given"
            )
        if not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer")
        self.random_ = check_random_state(self.random_state)
        if self.reward_type not in [
            "binary",
            "continuous",
        ]:
            raise ValueError(
                f"reward_type must be either 'binary' or 'continuous', but {self.reward_type} is given."
            )
        if self.reward_structure not in [
            "cascade_additive",
            "cascade_decay",
            "independent",
            "standard_additive",
            "standard_decay",
        ]:
            raise ValueError(
                f"reward_structure must be one of 'cascade_additive', 'cascade_decay', 'independent', 'standard_additive', or 'standard_decay', but {self.reward_structure} is given."
            )
        if self.decay_function not in ["exponential", "inverse"]:
            raise ValueError(
                f"decay_function must be either 'exponential' or 'inverse', but {self.decay_function} is given"
            )
        if self.click_model not in ["cascade", "pbm", None]:
            raise ValueError(
                f"click_model must be one of 'cascade', 'pbm', or None, but {self.click_model} is given."
            )
        # set exam_weight (slot-level examination probability).
        # When click_model is 'pbm', exam_weight is :math:`(1 / k)^{\\eta}`, where :math:`k` is the position.
        if self.click_model == "pbm":
            check_scalar(self.eta, name="eta", target_type=float, min_val=0.0)
            self.exam_weight = (1.0 / np.arange(1, self.len_list + 1)) ** self.eta
            self.attractiveness = np.ones(self.len_list, dtype=float)
        elif self.click_model == "cascade":
            check_scalar(self.eta, name="eta", target_type=float, min_val=0.0)
            self.attractiveness = (1.0 / np.arange(1, self.len_list + 1)) ** self.eta
            self.exam_weight = np.ones(self.len_list, dtype=float)
        else:
            self.attractiveness = np.ones(self.len_list, dtype=float)
            self.exam_weight = np.ones(self.len_list, dtype=float)
        if self.click_model is not None and self.reward_type == "continuous":
            raise ValueError(
                "continuous outcome cannot be generated when click_model is given"
            )
        if self.base_reward_function is not None:
            self.reward_function = action_interaction_reward_function
        if self.reward_structure in ["cascade_additive", "standard_additive"]:
            # generate additive action interaction weight matrix of (n_unique_action, n_unique_action)
            self.action_interaction_weight_matrix = generate_symmetric_matrix(
                n_unique_action=self.n_unique_action, random_state=self.random_state
            )
        else:
            # set decay function
            if self.decay_function == "exponential":
                self.decay_function = exponential_decay_function
            else:  # "inverse"
                self.decay_function = inverse_decay_function
            # generate decay action interaction weight matrix of (len_list, len_list)
            if self.reward_structure == "standard_decay":
                self.action_interaction_weight_matrix = (
                    self.obtain_standard_decay_action_interaction_weight_matrix(
                        self.len_list
                    )
                )
            elif self.reward_structure == "cascade_decay":
                self.action_interaction_weight_matrix = (
                    self.obtain_cascade_decay_action_interaction_weight_matrix(
                        self.len_list
                    )
                )
            else:
                self.action_interaction_weight_matrix = np.zeros(
                    (self.len_list, self.len_list)
                )
        if self.behavior_policy_function is None:
            self.uniform_behavior_policy = (
                np.ones(self.n_unique_action) / self.n_unique_action
            )
        if self.reward_type == "continuous":
            self.reward_min = 0
            self.reward_max = 1e10
            self.reward_std = 1.0
        # one-hot encoding representations characterizing each action
        self.action_context = np.eye(self.n_unique_action, dtype=int)

    def obtain_standard_decay_action_interaction_weight_matrix(
        self,
        len_list,
    ) -> np.ndarray:
        """Obtain action interaction weight matrix for standard decay reward structure (symmetric matrix)"""
        action_interaction_weight_matrix = np.identity(len_list)
        for position_ in np.arange(len_list):
            action_interaction_weight_matrix[:, position_] = -self.decay_function(
                np.abs(np.arange(len_list) - position_)
            )
            action_interaction_weight_matrix[position_, position_] = 0
        return action_interaction_weight_matrix

    def obtain_cascade_decay_action_interaction_weight_matrix(
        self,
        len_list,
    ) -> np.ndarray:
        """Obtain action interaction weight matrix for cascade decay reward structure (upper triangular matrix)"""
        action_interaction_weight_matrix = np.identity(len_list)
        for position_ in np.arange(len_list):
            action_interaction_weight_matrix[:, position_] = -self.decay_function(
                np.abs(np.arange(len_list) - position_)
            )
            for position_2 in np.arange(len_list):
                if position_ <= position_2:
                    action_interaction_weight_matrix[position_2, position_] = 0
        return action_interaction_weight_matrix

    def _calc_pscore_given_policy_logit(
        self, all_slate_actions: np.ndarray, policy_logit_i_: np.ndarray
    ) -> np.ndarray:
        """Calculate the propensity score of each of the possible slate actions given policy_logit.

        Parameters
        ------------
        all_slate_actions: array-like, (n_action, len_list)
            All possible slate actions.

        policy_logit_i_: array-like, (n_unique_action, )
            Logit values given context (:math:`x`), i.e., :math:`\\f: \\mathcal{X} \\rightarrow \\mathbb{R}^{\\mathcal{A}}`.

        Returns
        ------------
        pscores: array-like, (n_action, )
            Propensity scores of all the possible slate actions given policy_logit.

        """
        n_actions = len(all_slate_actions)
        unique_action_set_2d = np.tile(np.arange(self.n_unique_action), (n_actions, 1))
        pscores = np.ones(n_actions)
        for position_ in np.arange(self.len_list):
            action_index = np.where(
                unique_action_set_2d == all_slate_actions[:, position_][:, np.newaxis]
            )[1]
            pscores *= softmax(policy_logit_i_[unique_action_set_2d])[
                np.arange(n_actions), action_index
            ]
            # delete actions
            if position_ + 1 != self.len_list:
                mask = np.ones((n_actions, self.n_unique_action - position_))
                mask[np.arange(n_actions), action_index] = 0
                unique_action_set_2d = unique_action_set_2d[mask.astype(bool)].reshape(
                    (-1, self.n_unique_action - position_ - 1)
                )

        return pscores

    def _calc_pscore_given_policy_softmax(
        self, all_slate_actions: np.ndarray, policy_softmax_i_: np.ndarray
    ) -> np.ndarray:
        """Calculate the propensity score of each of the possible slate actions given policy_logit.

        Parameters
        ------------
        all_slate_actions: array-like, (n_action, len_list)
            All possible slate actions.

        policy_softmax_i_: array-like, (n_unique_action, )
            Policy values given context (:math:`x`), i.e., :math:`\\f: \\mathcal{X} \\rightarrow \\mathbb{R}^{\\mathcal{A}}`.

        Returns
        ------------
        pscores: array-like, (n_action, )
            Propensity scores of all the possible slate actions given policy_logit.

        """
        n_actions = len(all_slate_actions)
        unique_action_set_2d = np.tile(np.arange(self.n_unique_action), (n_actions, 1))
        pscores = np.ones(n_actions)
        for position_ in np.arange(self.len_list):
            action_index = np.where(
                unique_action_set_2d == all_slate_actions[:, position_][:, np.newaxis]
            )[1]
            score_ = policy_softmax_i_[unique_action_set_2d]
            pscores *= np.divide(score_, score_.sum(axis=1, keepdims=True))[
                np.arange(n_actions), action_index
            ]
            # delete actions
            if position_ + 1 != self.len_list:
                mask = np.ones((n_actions, self.n_unique_action - position_))
                mask[np.arange(n_actions), action_index] = 0
                unique_action_set_2d = unique_action_set_2d[mask.astype(bool)].reshape(
                    (-1, self.n_unique_action - position_ - 1)
                )

        return pscores

    def obtain_pscore_given_evaluation_policy_logit(
        self,
        action: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
    ):
        """Calculate the propensity score given evaluation policy logit.

        Parameters
        ------------
        action: array-like, (n_rounds * len_list, )
            Action chosen by behavior policy.

        evaluation_policy_logit_: array-like, (n_rounds, n_unique_action)
            Evaluation policy logit values by given context (:math:`x`), i.e., :math:`\\f: \\mathcal{X} \\rightarrow \\mathbb{R}^{\\mathcal{A}}`.

        return_pscore_item_position: bool, default=True
            A boolean parameter whether `pscore_item_position` is returned or not.
            When n_actions and len_list are large, giving True to this parameter may lead to a large computational time.

        clip_logit_value: Optional[float], default=None
            A float parameter to clip logit value (<= `700.`).
            When None is given, we calculate softmax values without clipping to obtain `pscore_item_position`.
            When a float value is given, we clip logit values to calculate softmax values to obtain `pscore_item_position`.
            When n_actions and len_list are large, giving None to this parameter may lead to a large computational time.

        """
        if not isinstance(action, np.ndarray) or action.ndim != 1:
            raise ValueError("action must be 1-dimensional ndarray")
        if (
            not isinstance(evaluation_policy_logit_, np.ndarray)
            or evaluation_policy_logit_.ndim != 2
        ):
            raise ValueError("evaluation_policy_logit_ must be 2-dimensional ndarray")
        if (
            len(action) / self.len_list != len(evaluation_policy_logit_)
            or evaluation_policy_logit_.shape[1] != self.n_unique_action
        ):
            raise ValueError(
                "the shape of action and evaluation_policy_logit_ must be (n_rounds * len_list, ) and (n_rounds, n_unique_action) respectively"
            )
        n_rounds = action.reshape((-1, self.len_list)).shape[0]
        pscore_cascade = np.zeros(n_rounds * self.len_list)
        pscore = np.zeros(n_rounds * self.len_list)
        if return_pscore_item_position:
            pscore_item_position = np.zeros(n_rounds * self.len_list)
            if not self.is_factorizable:
                enumerated_slate_actions = [
                    _
                    for _ in permutations(
                        np.arange(self.n_unique_action), self.len_list
                    )
                ]
                enumerated_slate_actions = np.array(enumerated_slate_actions)
        else:
            pscore_item_position = None
        if return_pscore_item_position and clip_logit_value is not None:
            check_scalar(
                clip_logit_value,
                name="clip_logit_value",
                target_type=(float),
                max_val=700.0,
            )
            evaluation_policy_softmax_ = np.exp(
                np.minimum(evaluation_policy_logit_, clip_logit_value)
            )
        for i in tqdm(
            np.arange(n_rounds),
            desc="[obtain_pscore_given_evaluation_policy_logit]",
            total=n_rounds,
        ):
            unique_action_set = np.arange(self.n_unique_action)
            score_ = softmax(evaluation_policy_logit_[i : i + 1])[0]
            pscore_i = 1.0
            for position_ in np.arange(self.len_list):
                action_ = action[i * self.len_list + position_]
                action_index_ = np.where(unique_action_set == action_)[0][0]
                # calculate joint pscore
                pscore_i *= score_[action_index_]
                pscore_cascade[i * self.len_list + position_] = pscore_i
                # update the pscore given the remaining items for nonfactorizable policy
                if not self.is_factorizable and position_ != self.len_list - 1:
                    unique_action_set = np.delete(
                        unique_action_set, unique_action_set == action_
                    )
                    score_ = softmax(
                        evaluation_policy_logit_[i : i + 1, unique_action_set]
                    )[0]
                # calculate pscore_item_position
                if return_pscore_item_position:
                    if position_ == 0:
                        pscore_item_position_i_l = pscore_i
                    elif self.is_factorizable:
                        pscore_item_position_i_l = score_[action_index_]
                    else:
                        if isinstance(clip_logit_value, float):
                            pscores = self._calc_pscore_given_policy_softmax(
                                all_slate_actions=enumerated_slate_actions,
                                policy_softmax_i_=evaluation_policy_softmax_[i],
                            )
                        else:
                            pscores = self._calc_pscore_given_policy_logit(
                                all_slate_actions=enumerated_slate_actions,
                                policy_logit_i_=evaluation_policy_logit_[i],
                            )
                        pscore_item_position_i_l = pscores[
                            enumerated_slate_actions[:, position_] == action_
                        ].sum()
                    pscore_item_position[
                        i * self.len_list + position_
                    ] = pscore_item_position_i_l
            # impute joint pscore
            start_idx = i * self.len_list
            end_idx = start_idx + self.len_list
            pscore[start_idx:end_idx] = pscore_i

        return pscore, pscore_item_position, pscore_cascade

    def sample_action_and_obtain_pscore(
        self,
        behavior_policy_logit_: np.ndarray,
        n_rounds: int,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Sample action and obtain the three variants of the propensity scores.

        Parameters
        ------------
        behavior_policy_logit_: array-like, shape (n_rounds, n_actions)
            Logit values given context (:math:`x`), i.e., :math:`\\f: \\mathcal{X} \\rightarrow \\mathbb{R}^{\\mathcal{A}}`.

        n_rounds: int
            Number of rounds for synthetic bandit feedback data.

        return_pscore_item_position: bool, default=True
            A boolean parameter whether `pscore_item_position` is returned or not.
            When n_actions and len_list are large, giving True to this parameter may lead to a large computational time.

        clip_logit_value: Optional[float], default=None
            A float parameter to clip logit value (<= `700.`).
            When None is given, we calculate softmax values without clipping to obtain `pscore_item_position`.
            When a float value is given, we clip logit values to calculate softmax values to obtain `pscore_item_position`.
            When n_actions and len_list are large, giving None to this parameter may lead to a large computational time.

        Returns
        ----------
        action: array-like, shape (n_rounds * len_list)
            Actions sampled by a behavior policy.
            Action list of slate `i` is stored in action[`i` * `len_list`: (`i + 1`) * `len_list`]

        pscore_cascade: array-like, shape (n_rounds * len_list)
            Joint action choice probabilities above the slot (:math:`k`) in each slate given context (:math:`x`).
            i.e., :math:`\\pi_k: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A}^{k})`.

        pscore: array-like, shape (n_rounds * len_list)
            Joint action choice probabilities of the slate given context (:math:`x`).
            i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A}^{\\text{len_list}})`.

        pscore_item_position: array-like, shape (n_rounds * len_list)
            Marginal action choice probabilities of each slot given context (:math:`x`).
            i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.

        """
        action = np.zeros(n_rounds * self.len_list, dtype=int)
        pscore_cascade = np.zeros(n_rounds * self.len_list)
        pscore = np.zeros(n_rounds * self.len_list)
        if return_pscore_item_position:
            pscore_item_position = np.zeros(n_rounds * self.len_list)
            if not self.is_factorizable and self.behavior_policy_function is not None:
                enumerated_slate_actions = [
                    _
                    for _ in permutations(
                        np.arange(self.n_unique_action), self.len_list
                    )
                ]
                enumerated_slate_actions = np.array(enumerated_slate_actions)
        else:
            pscore_item_position = None
        if return_pscore_item_position and clip_logit_value is not None:
            check_scalar(
                clip_logit_value,
                name="clip_logit_value",
                target_type=(float),
                max_val=700.0,
            )
            behavior_policy_softmax_ = np.exp(
                np.minimum(behavior_policy_logit_, clip_logit_value)
            )
        for i in tqdm(
            np.arange(n_rounds),
            desc="[sample_action_and_obtain_pscore]",
            total=n_rounds,
        ):
            unique_action_set = np.arange(self.n_unique_action)
            score_ = softmax(behavior_policy_logit_[i : i + 1, unique_action_set])[0]
            pscore_i = 1.0
            for position_ in np.arange(self.len_list):
                sampled_action = self.random_.choice(
                    unique_action_set, p=score_, replace=False
                )
                action[i * self.len_list + position_] = sampled_action
                sampled_action_index = np.where(unique_action_set == sampled_action)[0][
                    0
                ]
                # calculate joint pscore
                pscore_i *= score_[sampled_action_index]
                pscore_cascade[i * self.len_list + position_] = pscore_i
                # update the pscore given the remaining items for nonfactorizable behavior policy
                if not self.is_factorizable and position_ != self.len_list - 1:
                    unique_action_set = np.delete(
                        unique_action_set, unique_action_set == sampled_action
                    )
                    score_ = softmax(
                        behavior_policy_logit_[i : i + 1, unique_action_set]
                    )[0]
                # calculate pscore_item_position
                if return_pscore_item_position:
                    if self.behavior_policy_function is None:  # uniform random
                        pscore_item_position_i_l = 1 / self.n_unique_action
                    elif self.is_factorizable:
                        pscore_item_position_i_l = score_[sampled_action_index]
                    elif position_ == 0:
                        pscore_item_position_i_l = pscore_i
                    else:
                        if isinstance(clip_logit_value, float):
                            pscores = self._calc_pscore_given_policy_softmax(
                                all_slate_actions=enumerated_slate_actions,
                                policy_softmax_i_=behavior_policy_softmax_[i],
                            )
                        else:
                            pscores = self._calc_pscore_given_policy_logit(
                                all_slate_actions=enumerated_slate_actions,
                                policy_logit_i_=behavior_policy_logit_[i],
                            )
                        pscore_item_position_i_l = pscores[
                            enumerated_slate_actions[:, position_] == sampled_action
                        ].sum()
                    pscore_item_position[
                        i * self.len_list + position_
                    ] = pscore_item_position_i_l
            # impute joint pscore
            start_idx = i * self.len_list
            end_idx = start_idx + self.len_list
            pscore[start_idx:end_idx] = pscore_i

        return action, pscore_cascade, pscore, pscore_item_position

    def sample_contextfree_expected_reward(
        self, random_state: Optional[int] = None
    ) -> np.ndarray:
        """Sample expected reward for each action and slot from the uniform distribution

        Parameters
        -----------

        random_state: int, default=None
            Controls the random seed in sampling dataset.

        """
        random_ = check_random_state(random_state)
        return random_.uniform(size=(self.n_unique_action, self.len_list))

    def sample_reward_given_expected_reward(
        self, expected_reward_factual: np.ndarray
    ) -> np.ndarray:
        """Sample reward for each action and slot based on expected_reward_factual

        Parameters
        ------------
        expected_reward_factual: array-like, shape (n_rounds, len_list)
            Expected reward of factual actions given context.

        Returns
        ----------
        reward: array-like, shape (n_rounds, len_list)

        """
        expected_reward_factual *= self.exam_weight
        if self.reward_type == "binary":
            sampled_reward_list = list()
            discount_factors = np.ones(expected_reward_factual.shape[0])
            sampled_rewards_at_position = np.zeros(expected_reward_factual.shape[0])
            for position_ in np.arange(self.len_list):
                discount_factors *= sampled_rewards_at_position * self.attractiveness[
                    position_
                ] + (1 - sampled_rewards_at_position)
                expected_reward_factual_at_position = (
                    discount_factors * expected_reward_factual[:, position_]
                )
                sampled_rewards_at_position = self.random_.binomial(
                    n=1, p=expected_reward_factual_at_position
                )
                sampled_reward_list.append(sampled_rewards_at_position)
            reward = np.array(sampled_reward_list).T

        elif self.reward_type == "continuous":
            reward = np.zeros(expected_reward_factual.shape)
            for position_ in np.arange(self.len_list):
                mean = expected_reward_factual[:, position_]
                a = (self.reward_min - mean) / self.reward_std
                b = (self.reward_max - mean) / self.reward_std
                reward[:, position_] = truncnorm.rvs(
                    a=a,
                    b=b,
                    loc=mean,
                    scale=self.reward_std,
                    random_state=self.random_state,
                )
        else:
            raise NotImplementedError
        # return: array-like, shape (n_rounds, len_list)
        return reward

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        return_pscore_item_position: bool = True,
        clip_logit_value: Optional[float] = None,
    ) -> BanditFeedback:
        """Obtain batch logged bandit feedback.

        Parameters
        ----------
        n_rounds: int
            Number of rounds for synthetic bandit feedback data.

        return_pscore_item_position: bool, default=True
            A boolean parameter whether `pscore_item_position` is returned or not.
            When `n_unique_action` and `len_list` are large, this parameter should be set to False because of the computational time.

        clip_logit_value: Optional[float], default=None
            A float parameter to clip logit value.
            When None is given, we calculate softmax values without clipping to obtain `pscore_item_position`.
            When a float value is given, we clip logit values to calculate softmax values to obtain `pscore_item_position`.
            When n_actions and len_list are large, giving None to this parameter may lead to a large computational time.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Generated synthetic slate bandit feedback dataset.

        """
        if not isinstance(n_rounds, int) or n_rounds <= 0:
            raise ValueError(
                f"n_rounds must be a positive integer, but {n_rounds} is given"
            )

        context = self.random_.normal(size=(n_rounds, self.dim_context))
        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is None:
            behavior_policy_logit_ = np.tile(
                self.uniform_behavior_policy, (n_rounds, 1)
            )
        else:
            behavior_policy_logit_ = self.behavior_policy_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        # check the shape of behavior_policy_logit_
        if not (
            isinstance(behavior_policy_logit_, np.ndarray)
            and behavior_policy_logit_.shape == (n_rounds, self.n_unique_action)
        ):
            raise ValueError("behavior_policy_logit_ has an invalid shape")
        # sample actions and calculate the three variants of the propensity scores
        (
            action,
            pscore_cascade,
            pscore,
            pscore_item_position,
        ) = self.sample_action_and_obtain_pscore(
            behavior_policy_logit_=behavior_policy_logit_,
            n_rounds=n_rounds,
            return_pscore_item_position=return_pscore_item_position,
            clip_logit_value=clip_logit_value,
        )
        # sample expected reward factual
        if self.base_reward_function is None:
            expected_reward = self.sample_contextfree_expected_reward(
                random_state=self.random_state
            )
            expected_reward_tile = np.tile(expected_reward, (n_rounds, 1, 1))
            # action_2d: array-like, shape (n_rounds, len_list)
            action_2d = action.reshape((n_rounds, self.len_list))
            # expected_reward_factual: array-like, shape (n_rounds, len_list)
            expected_reward_factual = np.array(
                [
                    expected_reward_tile[
                        np.arange(n_rounds), action_2d[:, position_], position_
                    ]
                    for position_ in np.arange(self.len_list)
                ]
            ).T
        else:
            expected_reward_factual = self.reward_function(
                context=context,
                action_context=self.action_context,
                action=action,
                action_interaction_weight_matrix=self.action_interaction_weight_matrix,
                base_reward_function=self.base_reward_function,
                reward_type=self.reward_type,
                reward_structure=self.reward_structure,
                len_list=self.len_list,
                random_state=self.random_state,
            )
        # check the shape of expected_reward_factual
        if not (
            isinstance(expected_reward_factual, np.ndarray)
            and expected_reward_factual.shape == (n_rounds, self.len_list)
        ):
            raise ValueError("expected_reward_factual has an invalid shape")
        # sample reward
        reward = self.sample_reward_given_expected_reward(
            expected_reward_factual=expected_reward_factual
        )

        return dict(
            n_rounds=n_rounds,
            n_unique_action=self.n_unique_action,
            slate_id=np.repeat(np.arange(n_rounds), self.len_list),
            context=context,
            action_context=self.action_context,
            action=action,
            position=np.tile(np.arange(self.len_list), n_rounds),
            reward=reward.reshape(action.shape[0]),
            expected_reward_factual=expected_reward_factual.reshape(action.shape[0]),
            pscore_cascade=pscore_cascade,
            pscore=pscore,
            pscore_item_position=pscore_item_position,
        )

    def calc_on_policy_policy_value(
        self, reward: np.ndarray, slate_id: np.ndarray
    ) -> float:
        """Calculate the policy value of given reward and slate_id.

        Parameters
        -----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        slate_id: array-like, shape (<= n_rounds * len_list,)
            Slate ids of the logged bandit feedback.

        Returns
        ----------
        policy_value: float
            The on-policy policy value estimate of the behavior policy.

        """
        if not isinstance(reward, np.ndarray):
            raise ValueError("reward must be ndarray")
        if not isinstance(slate_id, np.ndarray):
            raise ValueError("slate_id must be ndarray")
        if reward.ndim != 1:
            raise ValueError(f"reward must be 1-dimensional, but is {reward.ndim}.")
        if slate_id.ndim != 1:
            raise ValueError(f"slate_id must be 1-dimensional, but is {slate_id.ndim}.")
        if reward.shape[0] != slate_id.shape[0]:
            raise ValueError(
                "the size of axis 0 of reward must be the same as that of slate_id"
            )

        return reward.sum() / np.unique(slate_id).shape[0]

    def calc_ground_truth_policy_value(
        self,
        context: np.ndarray,
        evaluation_policy_logit_: np.ndarray,
    ):
        """Calculate the ground-truth policy value of given evaluation policy logit and context

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each round (such as user information).

        evaluation_policy_logit_: array-like, shape (n_rounds, n_unique_action)
            Evaluation policy function generating logit value of each action in action space.

        """
        if (
            not isinstance(evaluation_policy_logit_, np.ndarray)
            or evaluation_policy_logit_.ndim != 2
        ):
            raise ValueError("evaluation_policy_logit_ must be 2-dimensional ndarray")
        if evaluation_policy_logit_.shape[1] != self.n_unique_action:
            raise ValueError(
                "the size of axis 1 of evaluation_policy_logit_ must be the same as n_unique_action"
            )
        if not isinstance(context, np.ndarray) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional ndarray")
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "the size of axis 1 of context must be the same as dim_context"
            )
        if evaluation_policy_logit_.shape[0] != context.shape[0]:
            raise ValueError(
                "the length of evaluation_policy_logit_ and context must be same"
            )

        if self.is_factorizable:
            enumerated_slate_actions = [
                _
                for _ in product(np.arange(self.n_unique_action), repeat=self.len_list)
            ]
        else:
            enumerated_slate_actions = [
                _ for _ in permutations(np.arange(self.n_unique_action), self.len_list)
            ]
        enumerated_slate_actions = np.array(enumerated_slate_actions).astype("int8")
        n_slate_actions = len(enumerated_slate_actions)
        n_rounds = len(evaluation_policy_logit_)

        pscores = []
        n_enumerated_slate_actions = len(enumerated_slate_actions)
        if self.is_factorizable:
            for action_list in tqdm(
                enumerated_slate_actions,
                desc="[calc_ground_truth_policy_value (pscore)]",
                total=n_enumerated_slate_actions,
            ):
                pscores.append(
                    softmax(evaluation_policy_logit_)[:, action_list].prod(1)
                )
            pscores = np.array(pscores).T
        else:
            for i in tqdm(
                np.arange(n_rounds),
                desc="[calc_ground_truth_policy_value (pscore)]",
                total=n_rounds,
            ):
                pscores.append(
                    self._calc_pscore_given_policy_logit(
                        all_slate_actions=enumerated_slate_actions,
                        policy_logit_i_=evaluation_policy_logit_[i],
                    )
                )
            pscores = np.array(pscores)

        # calculate expected slate-level reward for each combinatorial set of items (i.e., slate actions)
        if self.base_reward_function is None:
            expected_slot_reward = self.sample_contextfree_expected_reward(
                random_state=self.random_state
            )
            expected_slot_reward_tile = np.tile(
                expected_slot_reward, (n_rounds * n_slate_actions, 1, 1)
            )
            expected_slate_rewards = np.array(
                [
                    expected_slot_reward_tile[
                        np.arange(n_slate_actions) % n_slate_actions,
                        np.array(enumerated_slate_actions)[:, position_],
                        position_,
                    ]
                    for position_ in np.arange(self.len_list)
                ]
            ).T
            policy_value = (pscores * expected_slate_rewards.sum(axis=1)).sum()
        else:
            n_batch = (
                n_rounds * n_enumerated_slate_actions * self.len_list - 1
            ) // 10 ** 7 + 1
            batch_size = (n_rounds - 1) // n_batch + 1
            n_batch = (n_rounds - 1) // batch_size + 1

            policy_value = 0.0
            for batch_idx in tqdm(
                np.arange(n_batch),
                desc=f"[calc_ground_truth_policy_value (expected reward), batch_size={batch_size}]",
                total=n_batch,
            ):
                context_ = context[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]
                pscores_ = pscores[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]

                expected_slate_rewards_ = self.reward_function(
                    context=context_,
                    action_context=self.action_context,
                    action=enumerated_slate_actions.flatten(),
                    action_interaction_weight_matrix=self.action_interaction_weight_matrix,
                    base_reward_function=self.base_reward_function,
                    reward_type=self.reward_type,
                    reward_structure=self.reward_structure,
                    len_list=self.len_list,
                    is_enumerated=True,
                    random_state=self.random_state,
                )

                # click models based on expected reward
                expected_slate_rewards_ *= self.exam_weight
                if self.reward_type == "binary":
                    discount_factors = np.ones(expected_slate_rewards_.shape[0])
                    previous_slot_expected_reward = np.zeros(
                        expected_slate_rewards_.shape[0]
                    )
                    for position_ in np.arange(self.len_list):
                        discount_factors *= (
                            previous_slot_expected_reward
                            * self.attractiveness[position_]
                            + (1 - previous_slot_expected_reward)
                        )
                        expected_slate_rewards_[:, position_] = (
                            discount_factors * expected_slate_rewards_[:, position_]
                        )
                        previous_slot_expected_reward = expected_slate_rewards_[
                            :, position_
                        ]

                policy_value += (
                    pscores_.flatten() * expected_slate_rewards_.sum(axis=1)
                ).sum()
            policy_value /= n_rounds

        return policy_value

    def generate_evaluation_policy_pscore(
        self,
        evaluation_policy_type: str,
        context: np.ndarray,
        action: Optional[np.ndarray] = None,
        epsilon: Optional[float] = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the three variants of the propensity scores of synthetic evaluation policies (such as 'random', 'optimal', 'anti-optimal').

        Parameters
        -----------
        evaluation_policy_type: str
            Type of evaluation policy, which must be one of 'optimal', 'anti-optimal', or 'random'.
            When 'optimal' is given, we sort actions based on the base expected rewards (outputs of `base_reward_function`) and extract top-L actions (L=`len_list`) for each slate.
            When 'anti-optimal' is given, we sort actions based on the base expected rewards (outputs of `base_reward_function`) and extract bottom-L actions (L=`len_list`) for each slate.
            We calculate the three variants of the propensity scores (pscore, pscore_item_position, and pscore_cascade) of the epsilon-greedy policy when either 'optimal' or 'anti-optimal' is given.
            When 'random' is given, we calculate the three variants of the propensity scores of the uniform random policy.

        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each round (such as user information).

        action: array-like, shape (n_rounds * len_list,), default=None
            Actions sampled by a behavior policy.
            Action list of slate `i` is stored in action[`i` * `len_list`: (`i + 1`) * `len_list`].
            When evaluation_policy_type is 'random', this is unnecessary.

        epsilon: float, default=1.
            Exploration hyperparameter that must take value in the range of [0., 1.].
            When evaluation_policy_type is 'random', this is unnecessary.

        Returns
        ----------
        pscore: array-like, shape (n_unique_action * len_list)
            Joint action choice probabilities of the slate given context (:math:`x`).
            i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A}^{\\text{len_list}})`.

        pscore_item_position: array-like, shape (n_unique_action * len_list)
            Marginal action choice probabilities of each slot given context (:math:`x`).
            i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.

        pscore_cascade: array-like, shape (n_unique_action * len_list)
            Joint action choice probabilities above the slot (:math:`k`) in each slate given context (:math:`x`).
            i.e., :math:`\\pi_k: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A}^{k})`.

        """
        if evaluation_policy_type not in ["optimal", "anti-optimal", "random"]:
            raise ValueError(
                f"evaluation_policy_type must be 'optimal', 'anti-optimal', or 'random', but {evaluation_policy_type} is given"
            )
        if not isinstance(context, np.ndarray) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional ndarray")

        # [Caution]: OverflowError raises when integer division result is too large for a float
        if self.is_factorizable:
            random_pscore_cascade = (
                (np.ones((context.shape[0], self.len_list)) / self.n_unique_action)
                .cumprod(axis=1)
                .flatten()
            )
            random_pscore = np.ones(context.shape[0] * self.len_list) / (
                self.n_unique_action ** self.len_list
            )
        else:
            random_pscore_cascade = (
                1.0
                / np.tile(
                    np.arange(
                        self.n_unique_action, self.n_unique_action - self.len_list, -1
                    ),
                    (context.shape[0], 1),
                )
                .cumprod(axis=1)
                .flatten()
            )
            random_pscore = np.ones(context.shape[0] * self.len_list) / perm(
                self.n_unique_action, self.len_list
            )
        random_pscore_item_position = (
            np.ones(context.shape[0] * self.len_list) / self.n_unique_action
        )
        if evaluation_policy_type == "random":
            return random_pscore, random_pscore_item_position, random_pscore_cascade

        else:
            # base_expected_reward: array-like, shape (n_rounds, n_unique_action)
            base_expected_reward = self.base_reward_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
            if (
                not isinstance(action, np.ndarray)
                or action.ndim != 1
                or action.shape[0] != context.shape[0] * self.len_list
            ):
                raise ValueError(
                    "action must be 1-dimensional ndarray, shape (n_rounds * len_list)"
                )
            action_2d = action.reshape((context.shape[0], self.len_list))
            if context.shape[0] != action_2d.shape[0]:
                raise ValueError(
                    "the size of axis 0 of context must be the same as that of action_2d"
                )

            check_scalar(
                epsilon, name="epsilon", target_type=(float), min_val=0.0, max_val=1.0
            )
            if evaluation_policy_type == "optimal":
                sorted_actions = base_expected_reward.argsort(axis=1)[
                    :, : self.len_list
                ]
            else:
                sorted_actions = base_expected_reward.argsort(axis=1)[
                    :, -self.len_list :
                ]
            (
                pscore,
                pscore_item_position,
                pscore_cascade,
            ) = self._calc_epsilon_greedy_pscore(
                epsilon=epsilon,
                action_2d=action_2d,
                sorted_actions=sorted_actions,
                random_pscore=random_pscore,
                random_pscore_item_position=random_pscore_item_position,
                random_pscore_cascade=random_pscore_cascade,
            )
        return pscore, pscore_item_position, pscore_cascade

    def _calc_epsilon_greedy_pscore(
        self,
        epsilon: float,
        action_2d: np.ndarray,
        sorted_actions: np.ndarray,
        random_pscore: np.ndarray,
        random_pscore_item_position: np.ndarray,
        random_pscore_cascade: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the three variants of the propensity scores of synthetic evaluation policies based on the epsilon-greedy rule.

        Parameters
        -----------
        epsilon: float, default=1.
            Exploration hyperparameter that must take value in the range of [0., 1.].
            When evaluation_policy_type is 'random', this argument is unnecessary.

        action_2d: array-like, shape (n_rounds, len_list), default=None
            Actions sampled by a behavior policy.
            Action list of slate `i` is stored in action[`i`].
            When bandit_feedback is obtained by `obtain_batch_bandit_feedback`, we can obtain action_2d as follows: bandit_feedback["action"].reshape((n_rounds, len_list))
            When evaluation_policy_type is 'random', this argument is unnecessary.

        random_pscore: array-like, shape (n_unique_action * len_list, )
            Joint action choice probabilities of the slate given context (:math:`x`) when the evaluation policy is random.
            i.e., :math:`\\frac{1}{{}_{n} P _r)`, where :math:`n` is `n_unique_actions` and :math:`r` is `len_list`.

        random_pscore_item_position: array-like, shape (n_unique_action * len_list, )
            Marginal action choice probabilities of each slot given context (:math:`x`) when the evaluation policy is random.
            i.e., :math:`\\frac{1}{n)`, where :math:`n` is `n_unique_actions`.

        random_pscore_cascade: array-like, shape (n_unique_action * len_list, )
            Joint action choice probabilities above the slot (:math:`k`) in each slate given context (:math:`x`) when the evaluation policy is random.
            i.e., :math:`\\frac{1}{{}_{n} P _k)`, where :math:`n` is `n_unique_actions`.


        Returns
        ----------
        pscore: array-like, shape (n_unique_action * len_list)
            Joint action choice probabilities of the slate given context (:math:`x`).
            i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A}^{\\text{len_list}})`.

        pscore_item_position: array-like, shape (n_unique_action * len_list)
            Marginal action choice probabilities of each slot given context (:math:`x`).
            i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.

        pscore_cascade: array-like, shape (n_unique_action * len_list)
            Joint action choice probabilities above the slot (:math:`k`) in each slate given context (:math:`x`).
            i.e., :math:`\\pi_k: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A}^{k})`.

        """
        if not isinstance(action_2d, np.ndarray) or action_2d.ndim != 2:
            raise ValueError("action_2d must be 2-dimensional ndarray")
        if not self.is_factorizable and set(
            [np.unique(x).shape[0] for x in action_2d]
        ) != set([self.len_list]):
            raise ValueError(
                "when is_factorizable=False, actions of each slate must not be duplicated"
            )
        if self.is_factorizable:
            action_match_flg = (
                np.tile(sorted_actions[:, 0], (action_2d.shape[1], 1)).T == action_2d
            )
        else:
            action_match_flg = sorted_actions == action_2d
        pscore_flg = np.repeat(action_match_flg.all(axis=1), self.len_list)
        pscore_item_position_flg = action_match_flg.flatten()
        pscore_cascade_flg = action_match_flg.cumprod(axis=1).flatten()
        # calculate the three variants of the propensity scores based on the given epsilon value
        pscore = pscore_flg * (1 - epsilon) + epsilon * random_pscore
        pscore_item_position = (
            pscore_item_position_flg * (1 - epsilon)
            + epsilon * random_pscore_item_position
        )
        pscore_cascade = (
            pscore_cascade_flg * (1 - epsilon) + epsilon * random_pscore_cascade
        )
        return pscore, pscore_item_position, pscore_cascade


def generate_symmetric_matrix(n_unique_action: int, random_state: int) -> np.ndarray:
    """Generate symmetric matrix

    Parameters
    -----------

    n_unique_action: int (>= len_list)
        Number of actions.

    random_state: int
        Controls the random seed in sampling elements of matrix.

    Returns
    ---------
    symmetric_matrix: array-like, shape (n_unique_action, n_unique_action)
    """
    random_ = check_random_state(random_state)
    base_matrix = random_.normal(scale=5, size=(n_unique_action, n_unique_action))
    symmetric_matrix = (
        np.tril(base_matrix) + np.tril(base_matrix).T - np.diag(base_matrix.diagonal())
    )
    return symmetric_matrix


def action_interaction_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    action: np.ndarray,
    base_reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    reward_type: str,
    reward_structure: str,
    action_interaction_weight_matrix: np.ndarray,
    len_list: int,
    is_enumerated: bool = False,
    random_state: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """Reward function incorporating interactions among combinatorial action

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_unique_action, dim_action_context)
        Vector representation for each action.

    action: array-like, shape (n_rounds * len_list, ) or (len(enumerated_slate_actions) * len_list, )
        When is_enumerated=False, action corresponds to actions sampled by a (often behavior) policy.
        In this case, action list of slate `i` is stored in action[`i` * `len_list`: (`i + 1`) * `len_list`].
        When is_enumerated=True, action corresponds to the enumerated all possible combinatorial actions.

    base_reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function generating expected reward for each given action-context pair,
        i.e., :math:`\\mu: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None is set, context **independent** expected reward for each action will be
        sampled from the uniform distribution automatically.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary' is given, expected reward is transformed by logit function.

    reward_structure: str
        Reward structure.
        Must be either 'standard_additive', 'cascade_additive', 'standard_decay', or 'cascade_decay'.

    action_interaction_weight_matrix (`W`): array-like, shape (n_unique_action, n_unique_action) or (len_list, len_list)
        When using additive-type reward_structure, `W(i, j)` is the interaction term between action `i` and `j`.
        When using decay-type reward_structure, `W(i, j)` is the weight of how the expected reward of slot `i` affects that of slot `j`.

    len_list: int (> 1)
        Length of a list of actions recommended in each slate.
        When Open Bandit Dataset is used, 3 should be set.

    is_enumerate: bool
        Whether input action is corresponds to enumerated_slate_actions or not.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward_factual: array-like, shape (n_rounds, len_list)
        When reward_structure='standard_additive', :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j \\neq k} W(a(k), a(j)))`.
        When reward_structure='caccade_additive', :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j < k} W(a(k), a(j)))`.
        Otherwise, :math:`q_k(x, a) = g(g^{-1}(f(x, a(k))) + \\sum_{j \\neq k} g^{-1}(f(x, a(j))) * W(k, j)`

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")

    if not isinstance(action_context, np.ndarray) or action_context.ndim != 2:
        raise ValueError("action_context must be 2-dimensional ndarray")

    if not isinstance(action, np.ndarray) or action.ndim != 1:
        raise ValueError("action must be 1-dimensional ndarray")

    if is_enumerated and action.shape[0] % len_list != 0:
        raise ValueError(
            "the size of axis 0 of context times len_list must be multiple of that of action"
        )

    if not is_enumerated and action.shape[0] != len_list * context.shape[0]:
        raise ValueError(
            "the size of axis 0 of context times len_list must be same with that of action"
        )

    if reward_type not in [
        "binary",
        "continuous",
    ]:
        raise ValueError(
            f"reward_type must be either 'binary' or 'continuous', but {reward_type} is given."
        )

    if reward_structure not in [
        "standard_additive",
        "cascade_additive",
        "standard_decay",
        "cascade_decay",
        "independent",
    ]:
        raise ValueError(
            f"reward_structure must be either 'standard_additive', 'cascade_additive', 'standard_decay' or 'cascade_decay', but {reward_structure} is given."
        )

    is_additive = reward_structure in ["standard_additive", "cascade_additive"]
    is_cascade = reward_structure in ["cascade_additive", "cascade_decay"]

    if is_additive:
        if action_interaction_weight_matrix.shape != (
            action_context.shape[0],
            action_context.shape[0],
        ):
            raise ValueError(
                f"the shape of action_interaction_weight_matrix must be (action_context.shape[0], action_context.shape[0]), but {action_interaction_weight_matrix.shape}"
            )
    else:  # decay
        if action_interaction_weight_matrix.shape != (
            len_list,
            len_list,
        ):
            raise ValueError(
                f"the shape of action_interaction_weight_matrix must be (len_list, len_list), but {action_interaction_weight_matrix.shape}"
            )

    n_rounds = context.shape[0]
    # duplicate action
    if is_enumerated:
        action = np.tile(action, n_rounds)
    # action_2d: array-like, shape (n_rounds (* len(enumerated_action_list)), len_list)
    action_2d = action.reshape((-1, len_list)).astype("int8")
    n_enumerated_slate_actions = len(action) // n_rounds
    # expected_reward: array-like, shape (n_rounds, n_unique_action)
    expected_reward = base_reward_function(
        context=context, action_context=action_context, random_state=random_state
    )
    if reward_type == "binary":
        expected_reward = np.log(expected_reward / (1 - expected_reward)).astype(
            "float16"
        )
    expected_reward_factual = np.zeros_like(action_2d, dtype="float16")
    for position_ in np.arange(len_list):
        tmp_fixed_reward = expected_reward[
            np.arange(len(action_2d)) // n_enumerated_slate_actions,
            action_2d[:, position_],
        ]
        if reward_structure == "independent":
            continue
        elif is_additive:
            for position2_ in np.arange(len_list):
                if is_cascade:
                    if position_ <= position2_:
                        break
                elif position_ == position2_:
                    continue
                tmp_fixed_reward += action_interaction_weight_matrix[
                    action_2d[:, position_], action_2d[:, position2_]
                ]
        else:
            for position2_ in np.arange(len_list):
                if is_cascade:
                    if position_ <= position2_:
                        break
                elif position_ == position2_:
                    continue
                expected_reward_ = expected_reward[
                    np.arange(len(action_2d)) // n_enumerated_slate_actions,
                    action_2d[:, position2_],
                ]
                weight_ = action_interaction_weight_matrix[position_, position2_]
                tmp_fixed_reward += expected_reward_ * weight_
        expected_reward_factual[:, position_] = tmp_fixed_reward

    if reward_type == "binary":
        expected_reward_factual = sigmoid(expected_reward_factual)
    else:
        expected_reward_factual = np.clip(expected_reward_factual, 0, None)

    assert expected_reward_factual.shape == (
        action_2d.shape[0],
        len_list,
    ), f"response shape must be (n_rounds (* enumerated_slate_actions), len_list), but {expected_reward_factual.shape}"
    return expected_reward_factual


def linear_behavior_policy_logit(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
    tau: Union[int, float] = 1.0,
) -> np.ndarray:
    """Linear contextual behavior policy for synthetic slate bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_unique_action, dim_action_context)
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    tau: int or float, default=1.0
        A temperature parameter, controlling the randomness of the action choice.
        As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

    Returns
    ---------
    logit value: array-like, shape (n_rounds, n_unique_action)
        Logit given context (:math:`x`), i.e., :math:`\\f: \\mathcal{X} \\rightarrow \\mathbb{R}^{\\mathcal{A}}`.

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")

    if not isinstance(action_context, np.ndarray) or action_context.ndim != 2:
        raise ValueError("action_context must be 2-dimensional ndarray")

    check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    coef_ = random_.uniform(size=context.shape[1])
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        logits[:, d] = context @ coef_ + action_context[d] @ action_coef_

    return logits / tau


def exponential_decay_function(distance: np.ndarray) -> np.ndarray:
    """Calculate exponential discount factor for action interaction weight matrix.

    Parameters
    -----------
    distance: array-like, shape (len_list, )
        Distance between two slots.

    """
    if not isinstance(distance, np.ndarray) or distance.ndim != 1:
        raise ValueError("distance must be 1-dimensional ndarray")
    return np.exp(-distance)


def inverse_decay_function(distance: np.ndarray) -> np.ndarray:
    """Calculate inverse discount factor for action interaction weight matrix.

    Parameters
    -----------
    distance: array-like, shape (len_list, )
        Distance between two slots.

    """
    if not isinstance(distance, np.ndarray) or distance.ndim != 1:
        raise ValueError("distance must be 1-dimensional ndarray")
    return 1 / (distance + 1)
