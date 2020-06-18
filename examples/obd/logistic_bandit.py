from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from obp.policy import BaseContextualPolicy


@dataclass
class LogisticEpsilonGreedy(BaseContextualPolicy):
    """Logistic Epsilon Greedy."""
    epsilon: float = 0.
    policy_name: str = f'logistic_egreedy_{epsilon}'

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        self.model_list = [
            MiniBatchLogisticRegression(
                lambda_=self.lambda_list[i], alpha=self.alpha_list[i], dim=self.dim)
            for i in np.arange(self.n_actions)]
        self.reward_lists = [[] for i in np.arange(self.n_actions)]
        self.context_lists = [[] for i in np.arange(self.n_actions)]

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data."""
        if self.action_counts.min() == 0:
            return self.random_.choice(self.n_actions, size=self.len_list, replace=False)
        else:
            if self.random_.rand() > self.epsilon:
                theta = np.array([model.predict_proba(context) for model in self.model_list]).flatten()
                unsorted_max_arms = np.argpartition(-theta, self.len_list)[:self.len_list]
                return unsorted_max_arms[np.argsort(-theta[unsorted_max_arms])]
            else:
                return self.random_.choice(self.n_actions, size=self.len_list, replace=False)

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update parameters."""
        self.n_trial += 1
        self.action_counts[action] += 1
        self.reward_lists[action].append(reward)
        self.context_lists[action].append(context)
        if self.n_trial % self.batch_size == 0:
            for action, model in enumerate(self.model_list):
                if not len(self.reward_lists[action]) == 0:
                    model.fit(X=np.concatenate(self.context_lists[action], axis=0),
                              y=np.array(self.reward_lists[action]))
            self.reward_lists = [[] for i in np.arange(self.n_actions)]
            self.context_lists = [[] for i in np.arange(self.n_actions)]


@dataclass
class LogisticUCB(BaseContextualPolicy):
    """Logistic Upper Confidence Bound."""
    epsilon: float = 0.
    policy_name: str = f'logistic_ucb_{epsilon}'

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        self.model_list = [
            MiniBatchLogisticRegression(
                lambda_=self.lambda_list[i], alpha=self.alpha_list[i], dim=self.dim)
            for i in np.arange(self.n_actions)]
        self.reward_lists = [[] for i in np.arange(self.n_actions)]
        self.context_lists = [[] for i in np.arange(self.n_actions)]

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data."""
        if self.action_counts.min() == 0:
            return self.random_.choice(self.n_actions, size=self.len_list, replace=False)
        else:
            theta = np.array([model.predict_proba(context)
                              for model in self.model_list]).flatten()
            std = np.array([np.sqrt(np.sum((model._q ** (-1)) * (context ** 2)))
                            for model in self.model_list]).flatten()
            ucb_score = theta + self.epsilon * std
            unsorted_max_arms = np.argpartition(-ucb_score, self.len_list)[:self.len_list]
            return unsorted_max_arms[np.argsort(-ucb_score[unsorted_max_arms])]

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update parameters."""
        self.n_trial += 1
        self.action_counts[action] += 1
        self.reward_lists[action].append(reward)
        self.context_lists[action].append(context)
        if self.n_trial % self.batch_size == 0:
            for action, model in enumerate(self.model_list):
                if not len(self.reward_lists[action]) == 0:
                    model.fit(X=np.concatenate(self.context_lists[action], axis=0),
                              y=np.array(self.reward_lists[action]))
            self.reward_lists = [[] for i in np.arange(self.n_actions)]
            self.context_lists = [[] for i in np.arange(self.n_actions)]


@dataclass
class LogisticTS(BaseContextualPolicy):
    """Logistic Thompson Sampling."""
    policy_name: str = 'logistic_ts'

    def __init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        self.model_list = [
            MiniBatchLogisticRegression(
                lambda_=self.lambda_list[i], alpha=self.alpha_list[i], dim=self.dim)
            for i in np.arange(self.n_actions)]
        self.reward_lists = [[] for i in np.arange(self.n_actions)]
        self.context_lists = [[] for i in np.arange(self.n_actions)]

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select action for new data."""
        if self.action_counts.min() == 0:
            return self.random_.choice(self.n_actions, size=self.len_list, replace=False)
        else:
            theta = np.array([model.predict_proba_with_sampling(context)
                              for model in self.model_list]).flatten()
            unsorted_max_arms = np.argpartition(-theta, self.len_list)[:self.len_list]
            return unsorted_max_arms[np.argsort(-theta[unsorted_max_arms])]

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update parameters."""
        self.n_trial += 1
        self.action_counts[action] += 1
        self.reward_lists[action].append(reward)
        self.context_lists[action].append(context)
        if self.n_trial % self.batch_size == 0:
            for action, model in enumerate(self.model_list):
                if not len(self.reward_lists[action]) == 0:
                    model.fit(X=np.concatenate(self.context_lists[action], axis=0),
                              y=np.array(self.reward_lists[action]))
            self.reward_lists = [[] for i in np.arange(self.n_actions)]
            self.context_lists = [[] for i in np.arange(self.n_actions)]


@dataclass
class MiniBatchLogisticRegression:
    """MiniBatch Online Logistic Regression Model."""
    lambda_: float
    alpha: float
    dim: int

    def __post_init__(self) -> None:
        """Initialize Class."""
        self._m = np.zeros(self.dim)
        self._q = np.ones(self.dim) * self.lambda_

    def loss(self, w: np.ndarray, *args) -> float:
        """Calculate loss function."""
        X, y = args
        return 0.5 * (self._q * (w - self._m)).dot(w - self._m) + np.log(1 + np.exp(-y * w.dot(X.T))).sum()

    def grad(self, w: np.ndarray, *args) -> np.ndarray:
        """Calculate gradient."""
        X, y = args
        return self._q * (w - self._m) + (-1) * (((y * X.T) / (1. + np.exp(y * w.dot(X.T)))).T).sum(axis=0)

    def sample(self) -> np.ndarray:
        """Sample coefficient vector from the prior distribution."""
        return self.random_.normal(self._m, self.sd(), size=self.dim)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Update coefficient vector by the mini-batch data."""
        self._m = minimize(self.loss, self._m, args=(X, y), jac=self.grad, method="L-BFGS-B",
                           options={'maxiter': 20, 'disp': False}).x
        P = (1 + np.exp(1 + X.dot(self._m))) ** (-1)
        self._q = self._q + (P * (1 - P)).dot(X ** 2)

    def sd(self) -> np.ndarray:
        """Standard deviation for the coefficient vector."""
        return self.alpha * (self._q)**(-1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict extected probability by the expected coefficient."""
        return sigmoid(X.dot(self._m))

    def predict_proba_with_sampling(self, X: np.ndarray) -> np.ndarray:
        """Predict extected probability by the sampled coefficient."""
        return sigmoid(X.dot(self.sample()))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Calculate sigmoid function."""
    return 1. / (1. + np.exp(-x))
