# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic SLate Logged Bandit Feedback."""
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Union, List
from itertools import permutations

import numpy as np
from scipy.stats import truncnorm
from sklearn.utils import check_random_state, check_scalar
from tqdm import tqdm

from .base import BaseBanditDataset
from ..types import BanditFeedback
from ..utils import sigmoid, softmax


@dataclass
class SyntheticSlateBanditDataset(BaseBanditDataset):
    """Class for generating synthetic slate bandit dataset.

    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times,
    we have different bandit samples with the same setting.
    This can be used to estimate confidence intervals of the performances of OPE estimators.

    If None is set as `behavior_policy_function`, the synthetic data will be context-free bandit feedback.

    Parameters
    -----------
    n_actions: int
        Number of actions.

    dim_context: int, default=1
        Number of dimensions of context vectors.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary' is given, rewards are sampled from the Bernoulli distribution.
        When 'continuous' is given, rewards are sampled from the truncated Normal distribution with `scale=1`.
        The mean parameter of the reward distribution is determined by the `reward_function` specified by the next argument.

    reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function generating expected reward for each given action-context pair,
        i.e., :math:`\\mu: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None is set, context **independent** expected reward for each action will be
        sampled from the uniform distribution automatically.

    behavior_policy_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default=None
        Function generating probability distribution over action space,
        i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.
        If None is set, context **independent** uniform distribution will be used (uniform random behavior policy).

    random_state: int, default=12345
        Controls the random seed in sampling synthetic bandit dataset.

    dataset_name: str, default='synthetic_bandit_dataset'
        Name of the dataset.

    Examples
    ----------

    .. code-block:: python

        >>> import numpy as np
        >>> from obp.dataset import (
            SyntheticBanditDataset,
            linear_reward_function,
            linear_behavior_policy
        )

        # generate synthetic contextual bandit feedback with 10 actions.
        >>> dataset = SyntheticBanditDataset(
                n_actions=10,
                dim_context=5,
                reward_function=logistic_reward_function,
                behavior_policy=linear_behavior_policy,
                random_state=12345
            )
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=100000)
        >>> bandit_feedback
        {
            'n_rounds': 100000,
            'n_actions': 10,
            'context': array([[-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057],
                    [ 1.39340583,  0.09290788,  0.28174615,  0.76902257,  1.24643474],
                    [ 1.00718936, -1.29622111,  0.27499163,  0.22891288,  1.35291684],
                    ...,
                    [ 1.36946256,  0.58727761, -0.69296769, -0.27519988, -2.10289159],
                    [-0.27428715,  0.52635353,  1.02572168, -0.18486381,  0.72464834],
                    [-1.25579833, -1.42455203, -0.26361242,  0.27928604,  1.21015571]]),
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
            'action': array([7, 4, 0, ..., 7, 9, 6]),
            'position': None,
            'reward': array([0, 1, 1, ..., 0, 1, 0]),
            'expected_reward': array([[0.80210203, 0.73828559, 0.83199558, ..., 0.81190503, 0.70617705,
                    0.68985306],
                    [0.94119582, 0.93473317, 0.91345213, ..., 0.94140688, 0.93152449,
                    0.90132868],
                    [0.87248862, 0.67974991, 0.66965669, ..., 0.79229752, 0.82712978,
                    0.74923536],
                    ...,
                    [0.64856003, 0.38145901, 0.84476094, ..., 0.40962057, 0.77114661,
                    0.65752798],
                    [0.73208527, 0.82012699, 0.78161352, ..., 0.72361416, 0.8652249 ,
                    0.82571751],
                    [0.40348366, 0.24485417, 0.24037926, ..., 0.49613133, 0.30714854,
                    0.5527749 ]]),
            'pscore': array([0.05423855, 0.10339675, 0.09756788, ..., 0.05423855, 0.07250876,
                    0.14065505])
        }

    """

    n_actions: int
    len_list: int
    dim_context: int = 1
    reward_type: str = "binary"
    reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    random_state: int = 12345
    dataset_name: str = "synthetic_slate_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        if not isinstance(self.n_actions, int) or self.n_actions <= 1:
            raise ValueError(
                f"n_actions must be an integer larger than 1, but {self.n_actions} is given"
            )
        if (
            not isinstance(self.len_list, int)
            or self.len_list <= 1
            or self.len_list > self.n_actions
        ):
            raise ValueError(
                f"len_list must be an integer such that 1 < len_list <= n_actions, but {self.len_list} is given"
            )
        if not isinstance(self.dim_context, int) or self.dim_context <= 0:
            raise ValueError(
                f"dim_context must be a positive integer, but {self.dim_context} is given"
            )
        if self.reward_type not in [
            "binary",
            "continuous",
        ]:
            raise ValueError(
                f"reward_type must be either 'binary' or 'continuous, but {self.reward_type} is given.'"
            )
        if not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer")
        self.random_ = check_random_state(self.random_state)
        if self.behavior_policy_function is None:
            self.behavior_policy = np.ones(self.n_actions) / self.n_actions
        if self.reward_type == "continuous":
            self.reward_min = 0
            self.reward_max = 1e10
            self.reward_std = 1.0
        # one-hot encoding representations characterizing each action
        self.action_context = np.eye(self.n_actions, dtype=int)

    def get_marginal_pscore(
        self, perm: List[int], behavior_policy_logit_i_: np.ndarray
    ) -> float:
        action_set = np.arange(self.n_actions)
        pscore_ = 1.0
        for action in perm:
            score_ = softmax(behavior_policy_logit_i_[:, action_set])[0]
            action_index = np.where(action_set == action)[0][0]
            pscore_ *= score_[action_index]
            action_set = np.delete(action_set, action_set == action)
        return pscore_

    def sample_action(
        self,
        behavior_policy_logit_: np.ndarray,
        n_rounds: int,
        return_pscore_marginal: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        action = np.zeros(n_rounds * self.len_list, dtype=int)
        pscore_joint_above = np.zeros(n_rounds * self.len_list)
        pscore_joint_all = np.zeros(n_rounds * self.len_list)
        if return_pscore_marginal:
            pscore_marginal = np.zeros(n_rounds * self.len_list)
        else:
            pscore_marginal = None
        for i in tqdm(np.arange(n_rounds), desc="[sample_action]", total=n_rounds):
            action_set = np.arange(self.n_actions)
            pscore_i = 1.0
            for position_ in np.arange(self.len_list):
                score_ = softmax(behavior_policy_logit_[i : i + 1, action_set])[0]
                action_sampled = self.random_.choice(
                    action_set, p=score_, replace=False
                )
                action[i * self.len_list + position_] = action_sampled
                sampled_action_index = np.where(action_set == action_sampled)[0][0]
                # calculate joint pscore
                pscore_joint_above[i * self.len_list + position_] = (
                    pscore_i * score_[sampled_action_index]
                )
                pscore_i *= score_[sampled_action_index]
                action_set = np.delete(action_set, action_set == action_sampled)
                # calculate marginal pscore
                if return_pscore_marginal:
                    pscore_marginal_i_l = 0.0
                    for perm in permutations(range(self.n_actions), self.len_list):
                        if sampled_action_index not in perm:
                            continue
                        pscore_marginal_i_l += self.get_marginal_pscore(
                            perm=perm,
                            behavior_policy_logit_i_=behavior_policy_logit_[i : i + 1],
                        )
                    pscore_marginal[i * self.len_list + position_] = pscore_marginal_i_l

            # calculate joint pscore all
            start_idx = i * self.len_list
            end_idx = start_idx + self.len_list
            pscore_joint_all[start_idx:end_idx] = pscore_i

        return action, pscore_joint_above, pscore_joint_all, pscore_marginal

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
        tau: Union[int, float] = 1.0,
        return_pscore_marginal: bool = True,
    ) -> BanditFeedback:
        """Obtain batch logged bandit feedback.

        Parameters
        ----------
        n_rounds: int
            Number of rounds for synthetic bandit feedback data.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Generated synthetic bandit feedback dataset.

        """
        if not isinstance(n_rounds, int) or n_rounds <= 0:
            raise ValueError(
                f"n_rounds must be a positive integer, but {n_rounds} is given"
            )

        context = self.random_.normal(size=(n_rounds, self.dim_context))
        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is None:
            behavior_policy_logit_ = np.tile(self.behavior_policy, (n_rounds, 1))
        else:
            behavior_policy_logit_ = self.behavior_policy_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        (
            action,
            pscore_joint_above,
            pscore_joint_all,
            pscore_marginal,
        ) = self.sample_action(
            behavior_policy_logit_=behavior_policy_logit_,
            n_rounds=n_rounds,
            return_pscore_marginal=return_pscore_marginal,
        )
        # action_3d = np.identity(self.n_actions)[
        #     action.reshape((n_rounds, self.len_list))
        # ]

        reward = np.zeros(n_rounds * self.len_list, dtype=int)

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            impression_id=np.repeat(np.arange(n_rounds), self.len_list),
            context=context,
            action_context=self.action_context,
            action=action,
            position=np.tile(range(self.len_list), n_rounds),
            reward=reward,
            expected_reward=None,
            pscore_joint_above=pscore_joint_above,
            pscore_joint_all=pscore_joint_all,
            pscore_marginal=pscore_marginal,
        )


def logistic_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")

    if not isinstance(action_context, np.ndarray) or action_context.ndim != 2:
        raise ValueError("action_context must be 2-dimensional ndarray")

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    # each arm has different coefficient vectors
    coef_ = random_.uniform(size=(action_context.shape[0], context.shape[1]))
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        logits[:, d] = context @ coef_[d] + action_context[d] @ action_coef_

    return sigmoid(logits)


def linear_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear mean reward function for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")

    if not isinstance(action_context, np.ndarray) or action_context.ndim != 2:
        raise ValueError("action_context must be 2-dimensional ndarray")

    random_ = check_random_state(random_state)
    expected_reward = np.zeros((context.shape[0], action_context.shape[0]))
    # each arm has different coefficient vectors
    coef_ = random_.uniform(size=(action_context.shape[0], context.shape[1]))
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        expected_reward[:, d] = context @ coef_[d] + action_context[d] @ action_coef_

    return expected_reward


def linear_behavior_policy_logit(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
    tau: Union[int, float] = 1.0,
) -> np.ndarray:
    """Linear contextual behavior policy for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    tau: int or float, default=1.0
        A temperature parameter, controlling the randomness of the action choice.
        As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.


    Returns
    ---------
    behavior_policy: array-like, shape (n_rounds, n_actions)
        logit given context (:math:`x`), i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.

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
