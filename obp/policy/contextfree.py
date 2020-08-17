# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Context-Free Bandit Algorithms."""
from dataclasses import dataclass
from typing import Optional

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
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default: 1
        Number of samples used in a batch parameter update.

    random_state: int, default: None
        Controls the random seed in sampling actions.

    epsilon: float, default: 1.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    policy_name: str, default: f'egreedy_{epsilon}'.
        Name of bandit policy.

    """

    epsilon: float = 1.0
    policy_name: str = f"egreedy_{epsilon}"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert (
            0 <= self.epsilon <= 1
        ), f"epsilon must be between 0 and 1, but {self.epsilon} is given"
        super().__post_init__()

    def select_action(self) -> np.ndarray:
        """Select a list of actions.

        Returns
        ----------
        selected_actions: array-like shape (len_list, )
            List of selected actions.

        """
        if (self.random_.rand() > self.epsilon) and (self.action_counts.min() > 0):
            predicted_rewards = self.reward_counts / self.action_counts
            return predicted_rewards.argsort()[::-1][: self.len_list]
        else:
            return self.random_.choice(
                self.n_actions, size=self.len_list, replace=False
            )

    def update_params(self, action: int, reward: float) -> None:
        """Update policy parameters.

        Parameters
        ----------
        action: int
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.

        """
        self.n_trial += 1
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
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default: 1
        Number of samples used in a batch parameter update.

    random_state: int, default: None
        Controls the random seed in sampling actions.

    epsilon: float, default: 1.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    policy_name: str, default: 'random'.
        Name of bandit policy.

    """

    policy_name: str = "random"


@dataclass
class BernoulliTS(BaseContextFreePolicy):
    """Bernoulli Thompson Sampling Policy

    Parameters
    ----------
    n_actions: int
        Number of actions.

    len_list: int, default: 1
        Length of a list of recommended actions in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default: 1
        Number of samples used in a batch parameter update.

    random_state: int, default: None
        Controls the random seed in sampling actions.

    alpha: array-like, shape (n_actions, ), default: None
        Prior parameter vector for Beta distributions.

    beta: array-like, shape (n_actions, ), default: None
        Prior parameter vector for Beta distributions.

    policy_name: str, default: 'bts'
        Name of bandit policy.

    """

    alpha: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    policy_name: str = "bts"

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        self.alpha = np.ones(self.n_actions) if self.alpha is None else self.alpha
        self.beta = np.ones(self.n_actions) if self.beta is None else self.beta

    def select_action(self) -> np.ndarray:
        """Select a list of actions.

        Returns
        ----------
        selected_actions: array-like shape (len_list, )
            List of selected actions.

        """
        predicted_rewards = self.random_.beta(
            a=self.reward_counts + self.alpha,
            b=(self.action_counts - self.reward_counts) + self.beta,
        )
        return predicted_rewards.argsort()[::-1][: self.len_list]

    def update_params(self, action: int, reward: float) -> None:
        """Update policy parameters.

        Parameters
        ----------
        action: int
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.

        """
        self.n_trial += 1
        self.action_counts_temp[action] += 1
        self.reward_counts_temp[action] += reward
        if self.n_trial % self.batch_size == 0:
            self.action_counts = np.copy(self.action_counts_temp)
            self.reward_counts = np.copy(self.reward_counts_temp)
