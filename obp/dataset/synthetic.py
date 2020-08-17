# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic Logged Bandit Feedback."""
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from sklearn.utils import check_random_state

from .base import BaseSyntheticBanditDataset
from ..types import BanditFeedback
from ..utils import sigmoid, softmax


@dataclass
class SyntheticBanditDataset(BaseSyntheticBanditDataset):
    """Class for generating synthetic bandit dataset.

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

    dim_context: int, default: 1
        Number of dimensions of context vectors.

    dim_action_context: int, default: 1
        Number of dimensions of context vectors for each action.

    reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default: None
        Function generating expected reward with context and action context vectors,
        i.e., :math:`\\mu: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None is set, context **independent** expected reward for each action will be
        sampled from the uniform distribution automatically.

    behavior_policy_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default: None
        Function generating probability distribution over action space,
        i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.
        If None is set, context **independent** probability of choosing each action will be
        sampled from the dirichlet distribution automatically (context-free behavior policy).

    random_state: int, default: None
        Controls the random seed in sampling synthetic bandit dataset.

    dataset_name: str, default: 'synthetic_contextual_bandit_dataset'
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
                dim_action_context=5,
                reward_function=logistic_reward_function,
                behavior_policy=linear_behavior_policy,
                random_state=12345
            )
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=100000)
        >>> print(bandit_feedback)
        {'n_rounds': 100000,
         'n_actions': 10,
         'context': array([[ 0.06987669,  0.24667411, -0.0118616 ,  1.00481159,  1.32719461],
                [-0.91926156, -1.54910644,  0.0221846 ,  0.75836315, -0.66052433],
                [ 0.86258008, -0.0100319 ,  0.05000936,  0.67021559,  0.85296503],
                ...,
                [ 0.09658876,  2.03636863,  0.40584106, -0.49167468, -0.44993244],
                [-1.13892634, -1.71173775, -0.98117438,  1.84662775, -1.47738898],
                [ 1.19581374, -2.24630358,  0.25097774, -0.12573204, -1.07518047]]),
         'action': array([0, 1, 5, ..., 9, 1, 1]),
         'position': array([0, 0, 0, ..., 0, 0, 0]),
         'reward': array([1, 0, 1, ..., 1, 1, 0]),
         'expected_reward': array([[0.79484127, 0.98710467, 0.91364645, ..., 0.80883287, 0.0262742 ,
                0.86335842],
                [0.21316852, 0.63537277, 0.32594524, ..., 0.13998069, 0.00316771,
                0.55818704],
                [0.84340111, 0.98274578, 0.92609427, ..., 0.74362081, 0.03999977,
                0.83685006],
                ...,
                [0.66977957, 0.98321981, 0.96810184, ..., 0.47796594, 0.05266329,
                0.81784767],
                [0.12054673, 0.473379  , 0.2343796 , ..., 0.15433855, 0.00100676,
                0.56626301],
                [0.51637384, 0.58875776, 0.49215658, ..., 0.09978619, 0.01262061,
                0.46472179]]),
         'pscore': array([0.08443531, 0.42866938, 0.17304293, ..., 0.11438704, 0.42866938,
                0.42866938])}

    """

    n_actions: int
    dim_context: int = 1
    dim_action_context: int = 1
    reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    random_state: Optional[int] = None
    dataset_name: str = "synthetic_contextual_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert self.n_actions > 1 and isinstance(
            self.n_actions, int
        ), f"n_actions must be an integer larger than 1, but {self.n_actions} is given"
        assert self.dim_context > 0 and isinstance(
            self.dim_context, int
        ), f"dim_context must be a positive integer, but {self.dim_context} is given"
        assert self.dim_action_context > 0 and isinstance(
            self.dim_action_context, int
        ), f"dim_action_context must be a positive integer, but {self.dim_action_context} is given"

        self.random_ = check_random_state(self.random_state)
        if self.reward_function is None:
            self.expected_reward = self.sample_contextfree_expected_reward()
        if self.behavior_policy_function is None:
            self.behavior_policy = self.sample_contextfree_behavior_policy()
        self.action_context = self.sample_action_context()

    @property
    def len_list(self) -> int:
        """Length of recommendation lists."""
        return 1

    def sample_action_context(self) -> np.ndarray:
        """Sample action context vectors from the standard normal distribution."""
        return self.random_.normal(size=(self.n_actions, self.dim_action_context))

    def sample_contextfree_expected_reward(self) -> np.ndarray:
        """Sample expected reward for each action from the uniform distribution."""
        return self.random_.uniform(size=self.n_actions)

    def sample_contextfree_behavior_policy(self) -> np.ndarray:
        """Sample probability of choosing each action from the dirichlet distribution."""
        alpha = self.random_.uniform(size=self.n_actions)
        return self.random_.dirichlet(alpha=alpha)

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedback:
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
        context = self.random_.normal(size=(n_rounds, self.dim_context))

        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is None:
            action = self.random_.choice(
                np.arange(self.n_actions), p=self.behavior_policy, size=n_rounds
            )
            pscore = self.behavior_policy[action]
        else:
            behavior_policy_ = self.behavior_policy_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
            action = np.array(
                [
                    self.random_.choice(
                        np.arange(self.n_actions), p=behavior_policy_[i],
                    )
                    for i in np.arange(n_rounds)
                ]
            )
            pscore = behavior_policy_[np.arange(n_rounds), action]

        # sample reward for each round based on the reward function
        if self.reward_function is None:
            expected_reward_ = self.expected_reward
            reward = self.random_.binomial(n=1, p=expected_reward_[action])
        else:
            expected_reward_ = self.reward_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
            reward = self.random_.binomial(
                n=1, p=expected_reward_[np.arange(n_rounds), action]
            )
        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=context,
            action_context=self.action_context,
            action=action,
            position=np.zeros(n_rounds, dtype=int),
            reward=reward,
            expected_reward=expected_reward_,
            pscore=pscore,
        )


def logistic_reward_function(
    context: np.ndarray, action_context: np.ndarray, random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Context vectors characterizing each action.

    random_state: int, default: None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context and action context vectors,
        i.e., :math:`\\mu: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.

    """
    assert (
        isinstance(context, np.ndarray) and context.ndim == 2
    ), "context must be 2-dimensional ndarray"
    assert (
        isinstance(action_context, np.ndarray) and action_context.ndim == 2
    ), "action_context must be 2-dimensional ndarray"

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    # each arm has different coefficient vectors
    coef_ = random_.uniform(size=(action_context.shape[0], context.shape[1]))
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        logits[:, d] = context @ coef_[d] + action_context[d] @ action_coef_

    return sigmoid(logits)


def linear_behavior_policy(
    context: np.ndarray, action_context: np.ndarray, random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear contextual behavior policy for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Context vectors characterizing each action.

    random_state: int, default: None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    behavior_policy: array-like, shape (n_rounds, n_actions)
        Probability of choosing each action given context and action context vectors
        i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.

    """
    assert (
        isinstance(context, np.ndarray) and context.ndim == 2
    ), "context must be 2-dimensional ndarray"
    assert (
        isinstance(action_context, np.ndarray) and action_context.ndim == 2
    ), "action_context must be 2-dimensional ndarray"

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    coef_ = random_.uniform(size=context.shape[1])
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        logits[:, d] = context @ coef_ + action_context[d] @ action_coef_

    return softmax(logits)
