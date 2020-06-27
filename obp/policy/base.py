# Copyright (c) ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from sklearn.utils import check_random_state


@dataclass
class BaseContextFreePolicy(metaclass=ABCMeta):
    """Base class for context-free bandit policies.

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
        pass

    @abstractmethod
    def update_params(self, action: int, reward: float) -> None:
        """Update policy parameters."""
        pass


@dataclass
class BaseContextualPolicy(metaclass=ABCMeta):
    """Base class for contextual bandit policies.

    Parameters
    ----------
    dim: int
        Dimension of context vectors.

    n_actions: int
        Number of actions.

    len_list: int, default: 1
        Length of a list of recommended actions in each impression.
        When Open Bandit Dataset is used, 3 shouled be set.

    batch_size: int, default: 1
        Number of samples used in a batch parameter update.

    n_trial: int, default: 0
        Current number of trials in a bandit simulation.

    alpha_: float, default: 1.
        Prior parameter for the online logistic regression.

    lambda_: float, default: 1.
        Regularization hyperparameter for the online logistic regression.

    random_state: int, default: None
        Controls the random seed in sampling actions.

    policy_type: str, default: 'contextual'
        Type of bandit policy such as 'contextfree', 'contextual', and 'combinatorial'
    """
    dim: int
    n_actions: int
    len_list: int = 1
    batch_size: int = 1
    n_trial: int = 0
    alpha_: float = 1.
    lambda_: float = 1.
    random_state: Optional[int] = None
    policy_type: str = 'contextual'

    def __post_init__(self) -> None:
        """Initialize class."""
        self.random_ = check_random_state(self.random_state)
        self.alpha_list = self.alpha_ * np.ones(self.n_actions)
        self.lambda_list = self.lambda_ * np.ones(self.n_actions)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for i in np.arange(self.n_actions)]
        self.context_lists = [[] for i in np.arange(self.n_actions)]

    def initialize(self) -> None:
        """Initialize policy parameters."""
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for i in np.arange(self.n_actions)]
        self.context_lists = [[] for i in np.arange(self.n_actions)]

    @abstractmethod
    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select a list of actions."""
        pass

    @abstractmethod
    def update_params(self, action: float, reward: float, context: np.ndarray) -> None:
        """Update policy parameters."""
        pass


BanditPolicy = Union[BaseContextFreePolicy, BaseContextualPolicy]
