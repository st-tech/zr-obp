# Copyright (c) ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import BaseContextFreePolicy


@dataclass
class EpsilonGreedy(BaseContextFreePolicy):
    """Epsilon Greedy policy.

    Parameters
    ----------
    n_actions: int
        Number of actions.

    len_list: int, default: 1
        Length of a list of recommended actions in each impression.
        When Open Bandit Dataset is used, 3 shouled be set.

    batch_size: int, default: 1
        Number of samples used in a batch parameter update.

    n_trial: int, default: 0
        Current number of trials in a bandit simulation.

    random_state: int, default: None
        Controls the random seed in sampling actions.

    policy_type: str, default: 'contextfree'
        Type of the bandit policy such as 'contextfree', 'contextual', and 'combinatorial'.

    epsilon: float, default: 1.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    policy_name: str, default: f'egreedy_{epsilon}'.
        Name of bandit policy.
    """
    epsilon: float = 1.
    policy_name: str = f'egreedy_{epsilon}'

    assert 0 <= epsilon <= 1, f'epsilon must be in [0, 1], but {epsilon} is set.'

    def select_action(self) -> np.ndarray:
        """Select a list of actions.

        Returns
        ----------
        selected_actions: array
            List of selected actions.
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
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.
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
    n_actions: int
        Number of actions.

    len_list: int, default: 1
        Length of a list of recommended actions in each impression.
        When Open Bandit Dataset is used, 3 shouled be set.

    batch_size: int, default: 1
        Number of samples used in a batch parameter update.

    n_trial: int, default: 0
        Current number of trials in a bandit simulation.

    random_state: int, default: None
        Controls the random seed in sampling actions.

    policy_type: str, default: 'contextfree'
        Type of the bandit policy such as 'contextfree', 'contextual', and 'combinatorial'.

    epsilon: float, default: 1.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    policy_name: str, default: 'random'.
        Name of bandit policy.
    """
    policy_name: str = 'random'


@dataclass
class BernoulliTS(BaseContextFreePolicy):
    """Bernoulli Thompson Sampling Policy

    Parameters
    ----------
    Parameters
    ----------
    n_actions: int
        Number of actions.

    len_list: int, default: 1
        Length of a list of recommended actions in each impression.
        When Open Bandit Dataset is used, 3 shouled be set.

    batch_size: int, default: 1
        Number of samples used in a batch parameter update.

    n_trial: int, default: 0
        Current number of trials in a bandit simulation.

    random_state: int, default: None
        Controls the random seed in sampling actions.

    policy_type: str, default: 'contextfree'
        Type of the bandit policy such as 'contextfree', 'contextual', and 'combinatorial'.

    alpha: List[float]], default: None
        Prior parameter vector for Beta distributions.

    beta: List[float]], default: None
        Prior parameter vector for Beta distributions.

    policy_name: str, default: 'bts'
        Name of bandit policy.
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
        selected_actions: array
            List of selected actions.
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
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.
        """
        self.action_counts_temp[action] += 1
        self.reward_counts_temp[action] += reward
        if self.n_trial % self.batch_size == 0:
            self.action_counts = np.copy(self.action_counts_temp)
            self.reward_counts = np.copy(self.reward_counts_temp)
