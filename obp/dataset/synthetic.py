# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic Logged Bandit Feedback."""
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from scipy.stats import truncnorm
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

    dim_context: int, default=1
        Number of dimensions of context vectors.

    reward_type: str, default='binary'
        Type of reward variable, must be either 'binary' or 'continuous'.
        When 'binary' is given, rewards are sampled from the Bernoulli distribution.
        When 'continuous' is given, rewards are sampled from the truncated Normal distribution with `scale=1`.

    reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function generating expected reward with context and action context vectors,
        i.e., :math:`\\mu: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None is set, context **independent** expected reward for each action will be
        sampled from the uniform distribution automatically.

    behavior_policy_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default=None
        Function generating probability distribution over action space,
        i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.
        If None is set, context **independent** uniform distribution will be used (uniform random behavior policy).

    random_state: int, default=None
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
            'position': array([0, 0, 0, ..., 0, 0, 0]),
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
    dim_context: int = 1
    reward_type: str = "binary"
    reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    random_state: Optional[int] = None
    dataset_name: str = "synthetic_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert self.n_actions > 1 and isinstance(
            self.n_actions, int
        ), f"n_actions must be an integer larger than 1, but {self.n_actions} is given"
        assert self.dim_context > 0 and isinstance(
            self.dim_context, int
        ), f"dim_context must be a positive integer, but {self.dim_context} is given"
        assert self.reward_type in [
            "binary",
            "continuous",
        ], f"reward_type must be either 'binary' or 'continuous, but {self.reward_type} is given.'"

        self.random_ = check_random_state(self.random_state)
        if self.reward_function is None:
            self.expected_reward = self.sample_contextfree_expected_reward()
        if self.behavior_policy_function is None:
            self.behavior_policy = np.ones(self.n_actions) / self.n_actions
        # one-hot encoding representations characterizing each action
        self.action_context = np.eye(self.n_actions, dtype=int)

    @property
    def len_list(self) -> int:
        """Length of recommendation lists."""
        return 1

    def sample_contextfree_expected_reward(self) -> np.ndarray:
        """Sample expected reward for each action from the uniform distribution."""
        return self.random_.uniform(size=self.n_actions)

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
        assert n_rounds > 0 and isinstance(
            n_rounds, int
        ), f"n_rounds must be a positive integer, but {n_rounds} is given"

        context = self.random_.normal(size=(n_rounds, self.dim_context))
        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is None:
            behavior_policy_ = np.tile(self.behavior_policy, (n_rounds, 1))
            action = self.random_.choice(
                np.arange(self.n_actions), p=self.behavior_policy, size=n_rounds
            )
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
            expected_reward_ = np.tile(self.expected_reward, (n_rounds, 1))
        else:
            expected_reward_ = self.reward_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        expected_reward_factual = expected_reward_[np.arange(n_rounds), action]
        if self.reward_type == "binary":
            reward = self.random_.binomial(n=1, p=expected_reward_factual)
        elif self.reward_type == "continuous":
            min_, max_ = 0, 1e10
            mean, std = expected_reward_factual, 1.0
            a, b = (min_ - mean) / std, (max_ - mean) / std
            reward = truncnorm.rvs(
                a=a, b=b, loc=mean, scale=std, random_state=self.random_state
            )
            # correct expected_reward_, as we use truncated normal distribution here
            mean = expected_reward_
            a, b = (min_ - mean) / std, (max_ - mean) / std
            expected_reward_ = truncnorm.stats(
                a=a, b=b, loc=mean, scale=std, moments="m"
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
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

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


def linear_reward_function(
    context: np.ndarray, action_context: np.ndarray, random_state: Optional[int] = None,
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
    assert (
        isinstance(context, np.ndarray) and context.ndim == 2
    ), "context must be 2-dimensional ndarray"
    assert (
        isinstance(action_context, np.ndarray) and action_context.ndim == 2
    ), "action_context must be 2-dimensional ndarray"

    random_ = check_random_state(random_state)
    expected_reward = np.zeros((context.shape[0], action_context.shape[0]))
    # each arm has different coefficient vectors
    coef_ = random_.uniform(size=(action_context.shape[0], context.shape[1]))
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        expected_reward[:, d] = context @ coef_[d] + action_context[d] @ action_coef_

    return expected_reward


def linear_behavior_policy(
    context: np.ndarray, action_context: np.ndarray, random_state: Optional[int] = None,
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

    Returns
    ---------
    behavior_policy: array-like, shape (n_rounds, n_actions)
        Action choice probabilities given context (:math:`x`), i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.

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
