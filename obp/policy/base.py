# Copyright (c) ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Base Interfaces for Bandit Algorithms."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state

from ..ope import RegressionModel


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
    policy_type: str = "contextfree"

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
    alpha_: float = 1.0
    lambda_: float = 1.0
    random_state: Optional[int] = None
    policy_type: str = "contextual"

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


@dataclass
class BaseOffPolicyLearner(metaclass=ABCMeta):
    """Base Class for off-policy learner with standard OPE estimators.

    Note
    ------

    Parameters
    -----------
    base_model: ClassifierMixin
        Machine learning classifier to be used to create the decision making policy.

    Examples
    ----------

        .. code-block:: python

    Reference
    -----------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    base_model: ClassifierMixin

    def __post_init__(self) -> None:
        """Initialize class."""
        pass

    @abstractmethod
    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        **kwargs,
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
        return NotImplementedError

    def fit(
        self,
        context: np.ndarray,
        action: float,
        reward: float,
        pscore: Optional[np.ndarray] = None,
        action_context: Optional[np.ndarray] = None,
        regression_model: Optional[RegressionModel] = None,
    ) -> None:
        """Fits the offline bandit policy according to the given logged bandit feedback data.

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

        action_context: array-like, shape (n_actions, dim_action_context), default: None
            Context vectors used as input to predict the mean reward function.

        regression_model: Optional[RegressionModel], default: None
            Regression model that predicts the mean reward function :math:`E[Y | X, A]`.

        """
        X, sample_weight, y = self._create_train_data_for_opl(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            action_context=action_context,
            regression_model=regression_model,
        )
        self.base_model.fit(X=X, y=y, sample_weight=sample_weight)

    def predict(self, context: np.ndarray) -> None:
        """Predict best action for new data.

        Parameters
        -----------
        context: array like of shape (n_rounds_of_new_data, dim_context)
            Observed context vector for new data.

        Returns
        -----------
        pred: array like of shape (n_rounds_of_new_data,)
            Predicted best action for new data.

        """
        return self.base_model.predict(context)

    def predict_proba(self, context: np.ndarray) -> None:
        """Predict probabilities of each action being the best one for new data.

        Parameters
        -----------
        context: array like of shape (n_rounds_of_new_data, dim_context)
            Observed context vector for new data.

        Returns
        -----------
        pred_proba: array like of shape (n_rounds_of_new_data, n_actions)
            Probability estimates of each arm being the best one for new data.
            The returned estimates for all classes are ordered by the label of classes.

        """
        return self.base_model.predict_proba(context)


BanditPolicy = Union[BaseContextFreePolicy, BaseContextualPolicy]
