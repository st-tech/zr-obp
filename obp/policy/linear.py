# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Contextual Linear Bandit Algorithms."""
from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_scalar

from ..utils import check_array
from .base import BaseContextualPolicy


@dataclass
class BaseLinPolicy(BaseContextualPolicy):
    """Base class for contextual bandit policies using linear regression.

    Parameters
    ------------
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

    epsilon: float, default=0.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    """

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        self.theta_hat = np.zeros((self.dim, self.n_actions))
        self.A_inv = np.concatenate(
            [np.identity(self.dim) for _ in np.arange(self.n_actions)]
        ).reshape(self.n_actions, self.dim, self.dim)
        self.b = np.zeros((self.dim, self.n_actions))

        self.A_inv_temp = np.concatenate(
            [np.identity(self.dim) for _ in np.arange(self.n_actions)]
        ).reshape(self.n_actions, self.dim, self.dim)
        self.b_temp = np.zeros((self.dim, self.n_actions))

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update policy parameters.

        Parameters
        ------------
        action: int
            Selected action by the policy.

        reward: float
            Observed reward for the chosen action and position.

        context: array-like, shape (1, dim_context)
            Observed context vector.

        """
        self.n_trial += 1
        self.action_counts[action] += 1
        # update the inverse matrix by the Woodbury formula
        self.A_inv_temp[action] -= (
            self.A_inv_temp[action]
            @ context.T
            @ context
            @ self.A_inv_temp[action]
            / (1 + context @ self.A_inv_temp[action] @ context.T)[0][0]
        )
        self.b_temp[:, action] += reward * context.flatten()
        if self.n_trial % self.batch_size == 0:
            self.A_inv, self.b = (
                np.copy(self.A_inv_temp),
                np.copy(self.b_temp),
            )


@dataclass
class LinEpsilonGreedy(BaseLinPolicy):
    """Linear Epsilon Greedy.

    Parameters
    ------------
    dim: int
        Number of dimensions of context vectors.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    batch_size: int, default=1
        Number of samples used in a batch parameter update.

    n_trial: int, default=0
        Current number of trials in a bandit simulation.

    random_state: int, default=None
        Controls the random seed in sampling actions.

    epsilon: float, default=0.
        Exploration hyperparameter that must take value in the range of [0., 1.].

    References
    ------------
    L. Li, W. Chu, J. Langford, and E. Schapire.
    A contextual-bandit approach to personalized news article recommendation.
    In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.

    """

    epsilon: float = 0.0

    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0, max_val=1.0)
        self.policy_name = f"linear_epsilon_greedy_{self.epsilon}"

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
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")

        if self.random_.rand() > self.epsilon:
            self.theta_hat = np.concatenate(
                [
                    self.A_inv[i] @ self.b[:, i][:, np.newaxis]
                    for i in np.arange(self.n_actions)
                ],
                axis=1,
            )  # dim * n_actions
            predicted_rewards = (context @ self.theta_hat).flatten()
            return predicted_rewards.argsort()[::-1][: self.len_list]
        else:
            return self.random_.choice(
                self.n_actions, size=self.len_list, replace=False
            )


@dataclass
class LinUCB(BaseLinPolicy):
    """Linear Upper Confidence Bound.

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

    epsilon: float, default=0.
        Exploration hyperparameter that must be greater than or equal to 0.0.

    References
    --------------
    L. Li, W. Chu, J. Langford, and E. Schapire.
    A contextual-bandit approach to personalized news article recommendation.
    In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.

    """

    epsilon: float = 0.0

    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0)
        self.policy_name = f"linear_ucb_{self.epsilon}"

        super().__post_init__()

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data.

        Parameters
        ----------
        context: array
            Observed context vector.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.

        """
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")

        self.theta_hat = np.concatenate(
            [
                self.A_inv[i] @ self.b[:, i][:, np.newaxis]
                for i in np.arange(self.n_actions)
            ],
            axis=1,
        )  # dim * n_actions
        sigma_hat = np.concatenate(
            [
                np.sqrt(context @ self.A_inv[i] @ context.T)
                for i in np.arange(self.n_actions)
            ],
            axis=1,
        )  # 1 * n_actions
        ucb_scores = (context @ self.theta_hat + self.epsilon * sigma_hat).flatten()
        return ucb_scores.argsort()[::-1][: self.len_list]


@dataclass
class LinTS(BaseLinPolicy):
    """Linear Thompson Sampling.

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

    def __post_init__(self) -> None:
        """Initialize class."""
        self.policy_name = "linear_ts"

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
        self.theta_hat = np.concatenate(
            [
                self.A_inv[i] @ self.b[:, i][:, np.newaxis]
                for i in np.arange(self.n_actions)
            ],
            axis=1,
        )
        theta_sampled = np.concatenate(
            [
                self.random_.multivariate_normal(self.theta_hat[:, i], self.A_inv[i])[
                    :, np.newaxis
                ]
                for i in np.arange(self.n_actions)
            ],
            axis=1,
        )

        predicted_rewards = (context @ theta_sampled).flatten()
        return predicted_rewards.argsort()[::-1][: self.len_list]
