from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.utils import check_random_state


@dataclass
class BaseContextFreePolicy(metaclass=ABCMeta):
    """Base class for context-free bandit policies.

    Parameters
    ----------
    n_actions: int
        The number of actions.

    len_list: int, default: 1
        The length of a list of recommended actions in each impression.
        When Open Bandit Dataset is used, 3 shouled be set.

    batch_size: int, default: 1
        The number of samples used in a batch parameter update.

    n_trial: int, default: 0
        Current number of trials in a bandit simulation.

    random_state: int, default: None
        Controls the random seed in sampling actions.

    policy_type: str, default: 'contextfree'
        The type of the bandit policy.
    """
    n_actions: int
    len_list: int = 1
    batch_size: int = 1
    n_trial: int = 0
    random_state: Optional[int] = None
    policy_type: str = 'contextfree'

    def __post_init__(self) -> None:
        """Initialize Class."""
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.action_counts_temp = np.zeros(self.n_actions, dtype=int)
        self.reward_counts_temp = np.zeros(self.n_actions)
        self.reward_counts = np.zeros(self.n_actions)

    def initialize(self) -> None:
        """Initialize Parameters."""
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.action_counts_temp = np.zeros(self.n_actions, dtype=int)
        self.reward_counts_temp = np.zeros(self.n_actions)
        self.reward_counts = np.zeros(self.n_actions)

    @abstractmethod
    def select_action(self) -> np.ndarray:
        """Select a list of actions."""
        raise NotImplementedError

    @abstractmethod
    def update_params(self, action: int, reward: float) -> None:
        """Update policy parameters."""
        raise NotImplementedError


@dataclass
class EpsilonGreedy(BaseContextFreePolicy):
    """Epsilon Greedy policy.

    Parameters
    ----------
    epsilon: float, default: 1.
        The exploration hyperparameter that takes value in [0., 1.]
    """
    epsilon: float = 1.
    policy_name: str = f"egreedy_{epsilon}"

    assert 0 <= epsilon <= 1, f'epsilon must be in [0, 1], but {epsilon} is set.'

    def select_action(self) -> np.ndarray:
        """Select a list of actions.

        Returns
        ----------
        : array
        A list of selected actions.
        """
        self.n_trial += 1
        if self.random_.rand() > self.epsilon:
            unsorted_max_arms = np.argpartition(-self.reward_counts, self.len_list)[:self.len_list]
            return unsorted_max_arms[np.argsort(-self.reward_counts[unsorted_max_arms])]
        else:
            return self.random_.choice(self.n_actions, size=self.len_list, replace=False)

    def update_params(self, action: int, reward: float) -> None:
        """Update policy parameters.

        Parameters
        ----------
        action: int
            An selected action by the policy.

        reward: float
            An observed reward for the chosen action and position.
        """
        self.action_counts_temp[action] += 1
        n, old_reward = self.action_counts_temp[action], self.reward_counts_temp[action]
        self.reward_counts_temp[action] = (old_reward * (n - 1) / n) + (reward / n)
        if self.n_trial % self.batch_size == 0:
            self.action_counts = np.copy(self.action_counts_temp)
            self.reward_counts = np.copy(self.reward_counts_temp)


@dataclass
class Random(EpsilonGreedy):
    """Random policy

    Parameters
    ----------
    policy_name: str, default: 'bts'
        The name of the policy.
    """
    policy_name: str = 'random'


@dataclass
class BernoulliTS(BaseContextFreePolicy):
    """Bernoulli Thompson Sampling Policy

    Parameters
    ----------
    alpha: List[float]], default: None
        A prior parameter vector for Beta distributions.

    beta: List[float]], default: None
        A prior parameter vector for Beta distributions.

    policy_name: str, default: 'bts'
        The name of the policy.
    """
    alpha: Optional[List[float]] = None
    beta: Optional[List[float]] = None
    policy_name: str = 'bts'

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        self.alpha = np.ones(self.n_actions) if self.alpha is None else self.alpha
        self.beta = np.ones(self.n_actions) if self.beta is None else self.beta

    def select_action(self) -> np.ndarray:
        """Select a list of actions.

        Returns
        ----------
        : array
        A list of selected actions.
        """
        self.n_trial += 1
        theta = self.random_.beta(a=self.reward_counts + self.alpha,
                                  b=(self.action_counts - self.reward_counts) + self.beta)
        unsorted_max_arms = np.argpartition(-theta, self.len_list)[:self.len_list]
        return unsorted_max_arms[np.argsort(-theta[unsorted_max_arms])]

    def update_params(self, action: int, reward: float) -> None:
        """Update policy parameters.

        Parameters
        ----------
        action: int
            An selected action by the policy.

        reward: float
            An observed reward for the chosen action and position.
        """
        self.action_counts_temp[action] += 1
        self.reward_counts_temp[action] += reward
        if self.n_trial % self.batch_size == 0:
            self.action_counts = np.copy(self.action_counts_temp)
            self.reward_counts = np.copy(self.reward_counts_temp)
