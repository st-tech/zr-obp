# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Base Interfaces for Bandit Algorithms."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_random_state

from .policy_type import PolicyType


@dataclass
class BaseContextFreePolicy(metaclass=ABCMeta):
    """Base class for context-free bandit policies.

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

    """

    n_actions: int
    len_list: int = 1
    batch_size: int = 1
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize Class."""
        if not isinstance(self.n_actions, int) or self.n_actions <= 1:
            raise ValueError(
                f"n_actions must be an integer larger than 1, but {self.n_actions} is given"
            )

        if not isinstance(self.len_list, int) or self.len_list <= 0:
            raise ValueError(
                f"len_list must be a positive integer, but {self.len_list} is given"
            )

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, but {self.batch_size} is given"
            )

        if self.n_actions < self.len_list:
            raise ValueError(
                f"n_actions >= len_list should hold, but n_actions is {self.n_actions} and len_list is {self.len_list}"
            )

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
        Length of a list of actions recommended in each impression.
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
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError(f"dim must be a positive integer, but {self.dim} is given")

        if not isinstance(self.n_actions, int) or self.n_actions <= 1:
            raise ValueError(
                f"n_actions must be an integer larger than 1, but {self.n_actions} is given"
            )

        if not isinstance(self.len_list, int) or self.len_list <= 0:
            raise ValueError(
                f"len_list must be a positive integer, but {self.len_list} is given"
            )

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, but {self.batch_size} is given"
            )

        if self.n_actions < self.len_list:
            raise ValueError(
                f"n_actions >= len_list should hold, but n_actions is {self.n_actions} and len_list is {self.len_list}"
            )

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
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    """

    n_actions: int
    len_list: int = 1

    def __post_init__(self) -> None:
        """Initialize class."""
        if not isinstance(self.n_actions, int) or self.n_actions <= 1:
            raise ValueError(
                f"n_actions must be an integer larger than 1, but {self.n_actions} is given"
            )

        if not isinstance(self.len_list, int) or self.len_list <= 0:
            raise ValueError(
                f"len_list must be a positive integer, but {self.len_list} is given"
            )

        if self.n_actions < self.len_list:
            raise ValueError(
                f"n_actions >= len_list should hold, but n_actions is {self.n_actions} and len_list is {self.len_list}"
            )

    @property
    def policy_type(self) -> PolicyType:
        """Type of the bandit policy."""
        return PolicyType.OFFLINE

    @abstractmethod
    def fit(
        self,
    ) -> None:
        """Fits an offline bandit policy using the given logged bandit feedback data."""
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
            Action choices by a policy trained by calling the `fit` method.

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
        """Fits an offline bandit policy using the given logged bandit feedback data."""
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
            Action choices by a policy trained by calling the `fit` method.

        """
        raise NotImplementedError
