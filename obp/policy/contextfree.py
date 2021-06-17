# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Context-Free Bandit Algorithms."""
import os

# import pkg_resources
import yaml
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import BaseContextFreePolicy


# configurations to replicate the Bernoulli Thompson Sampling policy used in ZOZOTOWN production
prior_bts_file = os.path.join(os.path.dirname(__file__), "conf", "prior_bts.yaml")
with open(prior_bts_file, "rb") as f:
    production_prior_for_bts = yaml.safe_load(f)


@dataclass
class EpsilonGreedy(BaseContextFreePolicy):
    """Epsilon Greedy policy.

    Parameters
    ----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.

    epsilon: float, default=1.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    policy_name: str, default=f'egreedy_{epsilon}'.
        Name of bandit policy.

    """

    epsilon: float = 1.0

    def __post_init__(self) -> None:
        """Initialize Class."""
        if not 0 <= self.epsilon <= 1:
            raise ValueError(
                f"epsilon must be between 0 and 1, but {self.epsilon} is given"
            )
        self.policy_name = f"egreedy_{self.epsilon}"
        super().__post_init__()

    def select_action(self) -> np.ndarray:
        """Select a list of actions.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
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
        self.reward_counts_temp[action] += reward
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

    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.

    epsilon: float, default=1.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    policy_name: str, default='random'.
        Name of bandit policy.

    """

    policy_name: str = "random"

    def compute_batch_action_dist(
        self,
        n_rounds: int = 1,
    ) -> np.ndarray:
        """Compute the distribution over actions by Monte Carlo simulation.

        Parameters
        ----------
        n_rounds: int, default=1
            Number of rounds in the distribution over actions.
            (the size of the first axis of `action_dist`)

        Returns
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Probability estimates of each arm being the best one for each sample, action, and position.

        """
        action_dist = np.ones((n_rounds, self.n_actions, self.len_list)) * (
            1 / self.n_actions
        )
        return action_dist


@dataclass
class BernoulliTS(BaseContextFreePolicy):
    """Bernoulli Thompson Sampling Policy

    Parameters
    ----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.

    alpha: array-like, shape (n_actions, ), default=None
        Prior parameter vector for Beta distributions.

    beta: array-like, shape (n_actions, ), default=None
        Prior parameter vector for Beta distributions.

    is_zozotown_prior: bool, default=False
        Whether to use hyperparameters for the beta distribution used
        at the start of the data collection period in ZOZOTOWN.

    campaign: str, default=None
        One of the three possible campaigns considered in ZOZOTOWN, "all", "men", and "women".

    policy_name: str, default='bts'
        Name of bandit policy.

    """

    alpha: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    is_zozotown_prior: bool = False
    campaign: Optional[str] = None
    policy_name: str = "bts"

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        if self.is_zozotown_prior:
            if self.campaign is None:
                raise Exception(
                    "`campaign` must be specified when `is_zozotown_prior` is True."
                )
            self.alpha = production_prior_for_bts[self.campaign]["alpha"]
            self.beta = production_prior_for_bts[self.campaign]["beta"]
        else:
            self.alpha = np.ones(self.n_actions) if self.alpha is None else self.alpha
            self.beta = np.ones(self.n_actions) if self.beta is None else self.beta

    def select_action(self) -> np.ndarray:
        """Select a list of actions.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
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

    def compute_batch_action_dist(
        self,
        n_rounds: int = 1,
        n_sim: int = 100000,
    ) -> np.ndarray:
        """Compute the distribution over actions by Monte Carlo simulation.

        Parameters
        ----------
        n_rounds: int, default=1
            Number of rounds in the distribution over actions.
            (the size of the first axis of `action_dist`)

        n_sim: int, default=100000
            Number of simulations in the Monte Carlo simulation to compute the distribution over actions.

        Returns
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Probability estimates of each arm being the best one for each sample, action, and position.

        """
        action_count = np.zeros((self.n_actions, self.len_list))
        for _ in np.arange(n_sim):
            selected_actions = self.select_action()
            for pos in np.arange(self.len_list):
                action_count[selected_actions[pos], pos] += 1
        action_dist = np.tile(
            action_count / n_sim,
            (n_rounds, 1, 1),
        )
        return action_dist
