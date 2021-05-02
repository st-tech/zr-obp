# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Contextual Logistic Bandit Algorithms."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_random_state
from scipy.optimize import minimize

from .base import BaseContextualPolicy
from ..utils import sigmoid


@dataclass
class BaseLogisticPolicy(BaseContextualPolicy):
    """Base class for contextual bandit policies using logistic regression.

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

    alpha_: float, default=1.
        Prior parameter for the online logistic regression.

    lambda_: float, default=1.
        Regularization hyperparameter for the online logistic regression.

    random_state: int, default=None
        Controls the random seed in sampling actions.

    """

    alpha_: float = 1.0
    lambda_: float = 1.0

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        if not isinstance(self.alpha_, float) or self.alpha_ <= 0.0:
            raise ValueError(
                f"alpha_ should be a positive float, but {self.alpha_} is given"
            )

        if not isinstance(self.lambda_, float) or self.lambda_ <= 0.0:
            raise ValueError(
                f"lambda_ should be a positive float, but {self.lambda_} is given"
            )

        self.alpha_list = self.alpha_ * np.ones(self.n_actions)
        self.lambda_list = self.lambda_ * np.ones(self.n_actions)
        self.model_list = [
            MiniBatchLogisticRegression(
                lambda_=self.lambda_list[i],
                alpha=self.alpha_list[i],
                dim=self.dim,
                random_state=self.random_state,
            )
            for i in np.arange(self.n_actions)
        ]

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update policy parameters.

        Parameters
        ----------
        action: int
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.

        context: array-like, shape (1, dim_context)
            Observed context vector.

        """
        self.n_trial += 1
        self.action_counts[action] += 1
        self.reward_lists[action].append(reward)
        self.context_lists[action].append(context)
        if self.n_trial % self.batch_size == 0:
            for action, model in enumerate(self.model_list):
                if not len(self.reward_lists[action]) == 0:
                    model.fit(
                        X=np.concatenate(self.context_lists[action], axis=0),
                        y=np.array(self.reward_lists[action]),
                    )
            self.reward_lists = [[] for _ in np.arange(self.n_actions)]
            self.context_lists = [[] for _ in np.arange(self.n_actions)]


@dataclass
class LogisticEpsilonGreedy(BaseLogisticPolicy):
    """Logistic Epsilon Greedy.

    Parameters
    -----------
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

    alpha_: float, default=1.
        Prior parameter for the online logistic regression.

    lambda_: float, default=1.
        Regularization hyperparameter for the online logistic regression.

    epsilon: float, default=0.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    """

    epsilon: float = 0.0

    def __post_init__(self) -> None:
        """Initialize class."""
        if not 0 <= self.epsilon <= 1:
            raise ValueError(
                f"epsilon must be between 0 and 1, but {self.epsilon} is given"
            )
        self.policy_name = f"logistic_egreedy_{self.epsilon}"

        super().__post_init__()

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data.

        Parameters
        ----------
        context: array-like, shape (1, dim_context)
            Observed context vector.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.

        """
        if self.random_.rand() > self.epsilon:
            theta = np.array(
                [model.predict_proba(context) for model in self.model_list]
            ).flatten()
            return theta.argsort()[::-1][: self.len_list]
        else:
            return self.random_.choice(
                self.n_actions, size=self.len_list, replace=False
            )


@dataclass
class LogisticUCB(BaseLogisticPolicy):
    """Logistic Upper Confidence Bound.

    Parameters
    ------------
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

    alpha_: float, default=1.
        Prior parameter for the online logistic regression.

    lambda_: float, default=1.
        Regularization hyperparameter for the online logistic regression.

    epsilon: float, default=0.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    References
    ----------
    Lihong Li, Wei Chu, John Langford, and Robert E Schapire.
    "A Contextual-bandit Approach to Personalized News Article Recommendation," 2010.

    """

    epsilon: float = 0.0

    def __post_init__(self) -> None:
        """Initialize class."""
        if self.epsilon < 0:
            raise ValueError(
                f"epsilon must be positive scalar, but {self.epsilon} is given"
            )
        self.policy_name = f"logistic_ucb_{self.epsilon}"

        super().__post_init__()

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data.

        Parameters
        ------------
        context: array-like, shape (1, dim_context)
            Observed context vector.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.

        """
        theta = np.array(
            [model.predict_proba(context) for model in self.model_list]
        ).flatten()
        std = np.array(
            [
                np.sqrt(np.sum((model._q ** (-1)) * (context ** 2)))
                for model in self.model_list
            ]
        ).flatten()
        ucb_score = theta + self.epsilon * std
        return ucb_score.argsort()[::-1][: self.len_list]


@dataclass
class LogisticTS(BaseLogisticPolicy):
    """Logistic Thompson Sampling.

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

    alpha_: float, default=1.
        Prior parameter for the online logistic regression.

    lambda_: float, default=1.
        Regularization hyperparameter for the online logistic regression.

    References
    ----------
    Olivier Chapelle and Lihong Li.
    "An empirical evaluation of thompson sampling," 2011.

    """

    policy_name: str = "logistic_ts"

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data.

        Parameters
        ----------
        context: array-like, shape (1, dim_context)
            Observed context vector.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.

        """
        theta = np.array(
            [model.predict_proba_with_sampling(context) for model in self.model_list]
        ).flatten()
        return theta.argsort()[::-1][: self.len_list]


@dataclass
class MiniBatchLogisticRegression:
    """MiniBatch Online Logistic Regression Model."""

    lambda_: float
    alpha: float
    dim: int
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize Class."""
        self._m = np.zeros(self.dim)
        self._q = np.ones(self.dim) * self.lambda_
        self.random_ = check_random_state(self.random_state)

    def loss(self, w: np.ndarray, *args) -> float:
        """Calculate loss function."""
        X, y = args
        return (
            0.5 * (self._q * (w - self._m)).dot(w - self._m)
            + np.log(1 + np.exp(-y * w.dot(X.T))).sum()
        )

    def grad(self, w: np.ndarray, *args) -> np.ndarray:
        """Calculate gradient."""
        X, y = args
        return self._q * (w - self._m) + (-1) * (
            ((y * X.T) / (1.0 + np.exp(y * w.dot(X.T)))).T
        ).sum(axis=0)

    def sample(self) -> np.ndarray:
        """Sample coefficient vector from the prior distribution."""
        return self.random_.normal(self._m, self.sd(), size=self.dim)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Update coefficient vector by the mini-batch data."""
        self._m = minimize(
            self.loss,
            self._m,
            args=(X, y),
            jac=self.grad,
            method="L-BFGS-B",
            options={"maxiter": 20, "disp": False},
        ).x
        P = (1 + np.exp(1 + X.dot(self._m))) ** (-1)
        self._q = self._q + (P * (1 - P)).dot(X ** 2)

    def sd(self) -> np.ndarray:
        """Standard deviation for the coefficient vector."""
        return self.alpha * (self._q) ** (-1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict extected probability by the expected coefficient."""
        return sigmoid(X.dot(self._m))

    def predict_proba_with_sampling(self, X: np.ndarray) -> np.ndarray:
        """Predict extected probability by the sampled coefficient."""
        return sigmoid(X.dot(self.sample()))
