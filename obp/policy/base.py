# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Base Interfaces for Bandit Algorithms."""
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from .policy_type import PolicyType


@dataclass
class BaseContextFreePolicy(metaclass=ABCMeta):
    """Base class for context-free bandit policies.

    Parameters
    ----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.

    """

    n_actions: int
    len_list: int = 1
    batch_size: int = 1
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1, max_val=self.n_actions)
        check_scalar(self.batch_size, "batch_size", int, min_val=1)
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.action_counts_temp = np.zeros(self.n_actions, dtype=int)
        self.reward_counts_temp = np.zeros(self.n_actions)
        self.reward_counts = np.zeros(self.n_actions)

    @property
    def policy_type(self) -> PolicyType:
        """Type of the bandit policy."""
        return PolicyType.CONTEXT_FREE

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
class BaseContextualPolicy(metaclass=ABCMeta):
    """Base class for contextual bandit policies.

    Parameters
    ----------
    dim: int
        Number of dimensions of context vectors.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    random_state: int, default=None
        Controls the random seed in sampling actions.

    """

    dim: int
    n_actions: int
    len_list: int = 1
    batch_size: int = 1
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.dim, "dim", int, min_val=1)
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1, max_val=self.n_actions)
        check_scalar(self.batch_size, "batch_size", int, min_val=1)
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for _ in np.arange(self.n_actions)]
        self.context_lists = [[] for _ in np.arange(self.n_actions)]

    @property
    def policy_type(self) -> PolicyType:
        """Type of the bandit policy."""
        return PolicyType.CONTEXTUAL

    def initialize(self) -> None:
        """Initialize policy parameters."""
        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for _ in np.arange(self.n_actions)]
        self.context_lists = [[] for _ in np.arange(self.n_actions)]

    @abstractmethod
    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select a list of actions."""
        raise NotImplementedError

    @abstractmethod
    def update_params(self, action: float, reward: float, context: np.ndarray) -> None:
        """Update policy parameters."""
        raise NotImplementedError


@dataclass
class BaseOfflinePolicyLearner(metaclass=ABCMeta):
    """Base class for off-policy learners.

    Parameters
    -----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    """

    n_actions: int
    len_list: int = 1

    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1, max_val=self.n_actions)

    @property
    def policy_type(self) -> PolicyType:
        """Type of the bandit policy."""
        return PolicyType.OFFLINE

    @abstractmethod
    def fit(
        self,
    ) -> None:
        """Fits an offline bandit policy on the given logged bandit data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict best action for new data.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        action: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choices made by a policy trained by calling the `fit` method.

        """
        raise NotImplementedError


@dataclass
class BaseContinuousOfflinePolicyLearner(metaclass=ABCMeta):
    """Base class for off-policy learners for the continuous action setting."""

    @property
    def policy_type(self) -> PolicyType:
        """Type of the bandit policy."""
        return PolicyType.OFFLINE

    @abstractmethod
    def fit(
        self,
    ) -> None:
        """Fits an offline bandit policy on the given logged bandit data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict the best continuous action value for new data.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        action: array-like, shape (n_rounds_of_new_data,)
            Action choices made by a policy trained by calling the `fit` method.

        """
        raise NotImplementedError
