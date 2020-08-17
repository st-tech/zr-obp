# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Base Interfaces for Bandit Algorithms."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.base import ClassifierMixin, is_classifier
from sklearn.utils import check_random_state

from ..utils import check_bandit_feedback_inputs


@dataclass
class BaseContextFreePolicy(metaclass=ABCMeta):
    """Base class for context-free bandit policies.

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

    """

    n_actions: int
    len_list: int = 1
    batch_size: int = 1
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert self.n_actions > 1 and isinstance(
            self.n_actions, int
        ), f"n_actions must be an integer larger than 1, but {self.n_actions} is given"
        assert self.len_list > 0 and isinstance(
            self.len_list, int
        ), f"len_list must be a positive integer, but {self.len_list} is given"
        assert self.batch_size > 0 and isinstance(
            self.batch_size, int
        ), f"batch_size must be a positive integer, but {self.batch_size} is given"

        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.action_counts_temp = np.zeros(self.n_actions, dtype=int)
        self.reward_counts_temp = np.zeros(self.n_actions)
        self.reward_counts = np.zeros(self.n_actions)

    @property
    def policy_type(self) -> str:
        """Type of the bandit policy."""
        return "contextfree"

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

    len_list: int, default: 1
        Length of a list of recommended actions in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default: 1
        Number of samples used in a batch parameter update.

    alpha_: float, default: 1.
        Prior parameter for the online logistic regression.

    lambda_: float, default: 1.
        Regularization hyperparameter for the online logistic regression.

    random_state: int, default: None
        Controls the random seed in sampling actions.

    """

    dim: int
    n_actions: int
    len_list: int = 1
    batch_size: int = 1
    alpha_: float = 1.0
    lambda_: float = 1.0
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize class."""
        assert self.dim > 0 and isinstance(
            self.dim, int
        ), f"dim must be a positive integer, but {self.dim} is given"
        assert self.n_actions > 1 and isinstance(
            self.n_actions, int
        ), f"n_actions must be an integer larger than 1, but {self.n_actions} is given"
        assert self.len_list > 0 and isinstance(
            self.len_list, int
        ), f"len_list must be a positive integer, but {self.len_list} is given"
        assert self.batch_size > 0 and isinstance(
            self.batch_size, int
        ), f"batch_size must be a positive integer, but {self.batch_size} is given"

        self.n_trial = 0
        self.random_ = check_random_state(self.random_state)
        self.alpha_list = self.alpha_ * np.ones(self.n_actions)
        self.lambda_list = self.lambda_ * np.ones(self.n_actions)
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.reward_lists = [[] for _ in np.arange(self.n_actions)]
        self.context_lists = [[] for _ in np.arange(self.n_actions)]

    @property
    def policy_type(self) -> str:
        """Type of the bandit policy."""
        return "contextual"

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
class BaseOffPolicyLearner(metaclass=ABCMeta):
    """Base Class for off-policy learner with OPE estimators.

    Parameters
    -----------
    base_model: ClassifierMixin
        Machine learning classifier to be used to estimate the loss function
        for learning the decision making policy.

    Reference
    -----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    base_model: ClassifierMixin

    def __post_init__(self) -> None:
        """Initialize class."""
        assert is_classifier(self.base_model), "base_model must be a classifier."

    @property
    def policy_type(self) -> str:
        """Type of the bandit policy."""
        return "offline"

    @abstractmethod
    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create training data for off-policy learning.

        Parameters
        -----------
        context: array-like, shape (n_actions,)
            Context vectors in the given training logged bandit feedback.

        action: array-like, shape (n_actions,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        reward: array-like, shape (n_actions,)
            Observed rewards in the given training logged bandit feedback.

        pscore: Optional[np.ndarray], default: None
            Propensity scores, the probability of selecting each action by behavior policy,
            in the given training logged bandit feedback.

        Returns
        --------
        (X, sample_weight, y): Tuple[np.ndarray, np.ndarray, np.ndarray]
            Feature vectors, sample weights, and outcome for training the base machine learning model.

        """
        raise NotImplementedError

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
    ) -> None:
        """Fits the offline bandit policy according to the given logged bandit feedback data.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in the given training logged bandit feedback.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        reward: array-like, shape (n_rounds,)
            Observed rewards in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds,), default: None
            Propensity scores, the probability of selecting each action by behavior policy,
            in the given training logged bandit feedback.

        """
        check_bandit_feedback_inputs(
            context=context, action=action, reward=reward, pscore=pscore,
        )
        X, sample_weight, y = self._create_train_data_for_opl(
            context=context, action=action, reward=reward, pscore=pscore,
        )
        self.base_model.fit(X=X, y=y, sample_weight=sample_weight)

    def predict(self, context: np.ndarray) -> None:
        """Predict best action for new data.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Observed context vector for new data.

        Returns
        -----------
        pred: array-like, shape (n_rounds_of_new_data, 1)
            Predicted best action for new data.

        """
        assert (
            isinstance(context, np.ndarray) and context.ndim == 2
        ), "context must be 2-dimensional ndarray"
        predicted_actions = self.base_model.predict(context)

        return np.expand_dims(predicted_actions, 1)

    def predict_proba(self, context: np.ndarray) -> None:
        """Predict probabilities of each action being the best one for new data.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Observed context vector for new data.

        Returns
        -----------
        pred_proba: array-like, shape (n_rounds_of_new_data, n_actions)
            Probability estimates of each arm being the best one for new data.
            The returned estimates for all classes are ordered by the label of classes.

        """
        assert (
            isinstance(context, np.ndarray) and context.ndim == 2
        ), "context must be 2-dimensional ndarray"

        return self.base_model.predict_proba(context)
