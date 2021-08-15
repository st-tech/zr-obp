# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic Continuous Logged Bandit Feedback."""
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from scipy.stats import uniform, truncnorm
from sklearn.utils import check_random_state, check_scalar

from .base import BaseBanditDataset
from ..types import BanditFeedback


@dataclass
class SyntheticContinuousBanditDataset(BaseBanditDataset):
    """Class for generating synthetic continuous bandit dataset.

    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times, we have different bandit samples with the same setting.
    This can be used to estimate confidence intervals of the performances of OPE estimators for continuous actions.
    If None is set as `behavior_policy_function`, the synthetic data will be context-free bandit feedback.

    Parameters
    -----------
    dim_context: int, default=1
        Number of dimensions of context vectors.

    action_noise: float, default=1.0
        Standard deviation of the Gaussian noise on the continuous action variable.

    reward_noise: float, default=1.0
        Standard deviation of the Gaussian noise on the reward variable.

    min_action_value: float, default=-np.inf
        A minimum possible continuous action value.

    max_action_value: float, default=np.inf
        A maximum possible continuous action value.

    reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function generating expected reward for each given action-context pair,
        i.e., :math:`\\mu: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None is set, context **independent** expected reward for each action will be
        sampled from the uniform distribution automatically.

    behavior_policy_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default=None
        Function generating the propensity score of continuous actions,
        i.e., :math:`\\f: \\mathcal{X} \\rightarrow \\mathbb{R}^{\\mathcal{A}}`.
        If None is set, context **independent** uniform distribution will be used (uniform behavior policy).

    random_state: int, default=12345
        Controls the random seed in sampling synthetic slate bandit dataset.

    dataset_name: str, default='synthetic_slate_bandit_dataset'
        Name of the dataset.

    Examples
    ----------

    .. code-block:: python

        >>> from obp.dataset import (
            SyntheticContinuousBanditDataset,
            linear_reward_funcion_continuous,
            linear_behavior_policy_continuous,
        )
        >>> dataset = SyntheticContinuousBanditDataset(
                dim_context=5,
                min_action_value=1,
                max_action_value=10,
                reward_function=linear_reward_funcion_continuous,
                behavior_policy_function=linear_behavior_policy_continuous,
                random_state=12345,
            )
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=10000)
        >>> bandit_feedback

        {
            'n_rounds': 10000,
            'context': array([[-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057],
                    [ 1.39340583,  0.09290788,  0.28174615,  0.76902257,  1.24643474],
                    [ 1.00718936, -1.29622111,  0.27499163,  0.22891288,  1.35291684],
                    ...,
                    [-1.27028221,  0.80914602, -0.45084222,  0.47179511,  1.89401115],
                    [-0.68890924,  0.08857502, -0.56359347, -0.41135069,  0.65157486],
                    [ 0.51204121,  0.65384817, -1.98849253, -2.14429131, -0.34186901]]),
            'action': array([7.15163752, 2.22523458, 1.80661079, ..., 3.23401871, 2.36257676,
                    3.46584587]),
            'reward': array([2.23806215, 3.04770578, 1.64975454, ..., 1.75709223, 1.07265021,
                    2.4478468 ]),
            'pscore': array([0.13484565, 0.39339631, 0.32859093, ..., 0.04650679, 0.34450074,
                    0.31665289]),
            'position': None,
            'expected_reward': array([3.01472331, 1.25381652, 0.9098273 , ..., 1.75787986, 1.04337996,
                    2.32619881])
        }

    """

    dim_context: int = 1
    action_noise: float = 1.0
    reward_noise: float = 1.0
    min_action_value: float = -np.inf
    max_action_value: float = np.inf
    reward_function: Optional[
        Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ]
    ] = None
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    random_state: int = 12345
    dataset_name: str = "synthetic_continuous_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.dim_context, name="dim_context", target_type=int, min_val=1)
        check_scalar(
            self.action_noise, name="action_noise", target_type=(int, float), min_val=0
        )
        check_scalar(
            self.reward_noise, name="reward_noise", target_type=(int, float), min_val=0
        )
        check_scalar(
            self.min_action_value, name="min_action_value", target_type=(int, float)
        )
        check_scalar(
            self.max_action_value, name="max_action_value", target_type=(int, float)
        )
        if self.max_action_value <= self.min_action_value:
            raise ValueError(
                "`max_action_value` must be larger than `min_action_value`"
            )
        if self.random_state is None:
            raise ValueError("random_state must be given")
        self.random_ = check_random_state(self.random_state)

    def _contextfree_reward_function(self, action: np.ndarray) -> np.ndarray:
        """
        Calculate context-free expected rewards given only continuous action values.
        This is just an example synthetic (expected) reward function.
        """
        return 2 * np.power(action, 1.5) - (5 * action)

    def obtain_batch_bandit_feedback(
        self,
        n_rounds: int,
    ) -> BanditFeedback:
        """Obtain batch logged bandit feedback.

        Parameters
        ----------
        n_rounds: int
            Number of rounds for synthetic bandit feedback data.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Generated synthetic bandit feedback dataset with continuous actions.

        """
        check_scalar(n_rounds, name="n_rounds", target_type=int, min_val=1)

        context = self.random_.normal(size=(n_rounds, self.dim_context))
        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is not None:
            expected_action_values = self.behavior_policy_function(
                context=context,
                random_state=self.random_state,
            )
            a = (self.min_action_value - expected_action_values) / self.action_noise
            b = (self.max_action_value - expected_action_values) / self.action_noise
            action = truncnorm.rvs(
                a,
                b,
                loc=expected_action_values,
                scale=self.action_noise,
                random_state=self.random_state,
            )
            pscore = truncnorm.pdf(
                action, a, b, loc=expected_action_values, scale=self.action_noise
            )
        else:
            action = uniform.rvs(
                loc=self.min_action_value,
                scale=(self.max_action_value - self.min_action_value),
                size=n_rounds,
                random_state=self.random_state,
            )
            pscore = uniform.pdf(
                action,
                loc=self.min_action_value,
                scale=(self.max_action_value - self.min_action_value),
            )

        if self.reward_function is None:
            expected_reward_ = self._contextfree_reward_function(action=action)
        else:
            expected_reward_ = self.reward_function(
                context=context, action=action, random_state=self.random_state
            )
        reward = expected_reward_ + self.random_.normal(
            scale=self.reward_noise, size=n_rounds
        )

        return dict(
            n_rounds=n_rounds,
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=None,  # position is irrelevant for continuous action data
            expected_reward=expected_reward_,
        )

    def calc_ground_truth_policy_value(
        self,
        context: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Calculate the policy value of the action sequence.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_test_data, dim_context)
            Context vectors of test data.

        action: array-like, shape (n_rounds_of_test_data,)
            Continuous action values for test data predicted by the (deterministic) evaluation policy, i.e., :math:`\\pi_e(x_t)`.

        Returns
        ----------
        policy_value: float
            The policy value of the evaluation policy on the given test bandit feedback data.

        """
        if not isinstance(context, np.ndarray) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional ndarray")
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "the size of axis 1 of context must be the same as dim_context"
            )
        if not isinstance(action, np.ndarray) or action.ndim != 1:
            raise ValueError("action must be 1-dimensional ndarray")
        if context.shape[0] != action.shape[0]:
            raise ValueError(
                "the size of axis 0 of context must be the same as that of action"
            )

        if self.reward_function is None:
            return self._contextfree_reward_function(action=action).mean()
        else:
            return self.reward_function(
                context=context, action=action, random_state=self.random_state
            ).mean()


# some functions to generate synthetic bandit feedback with continuous actions


def linear_reward_funcion_continuous(
    context: np.ndarray,
    action: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear reward function to generate synthetic continuous bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action: array-like, shape (n_rounds,)
        Continuous action values.

    random_state: int, default=None
        Controls the random seed in sampling parameters.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds,)
        Expected reward given context (:math:`x`) and continuous action (:math:`a`).

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")
    if not isinstance(action, np.ndarray) or action.ndim != 1:
        raise ValueError("action must be 1-dimensional ndarray")
    if context.shape[0] != action.shape[0]:
        raise ValueError(
            "the size of axis 0 of context must be the same as that of action"
        )

    random_ = check_random_state(random_state)
    coef_ = random_.normal(size=context.shape[1])
    pow_, bias = random_.uniform(size=2)
    return (np.abs(context @ coef_ - action) ** pow_) + bias


def quadratic_reward_funcion_continuous(
    context: np.ndarray,
    action: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Quadratic reward function to generate synthetic continuous bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action: array-like, shape (n_rounds,)
        Continuous action values.

    random_state: int, default=None
        Controls the random seed in sampling parameters.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds,)
        Expected reward given context (:math:`x`) and continuous action (:math:`a`).

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")
    if not isinstance(action, np.ndarray) or action.ndim != 1:
        raise ValueError("action must be 1-dimensional ndarray")
    if context.shape[0] != action.shape[0]:
        raise ValueError(
            "the size of axis 0 of context must be the same as that of action"
        )

    random_ = check_random_state(random_state)
    coef_x = random_.normal(size=context.shape[1])
    coef_x_a = random_.normal(size=context.shape[1])
    coef_x_a_squared = random_.normal(size=context.shape[1])
    coef_a = random_.normal(size=1)

    expected_reward = (coef_a * action) * (context @ coef_x)
    expected_reward += (context @ coef_x_a) * action
    expected_reward += (action - context @ coef_x_a_squared) ** 2
    return expected_reward


def linear_behavior_policy_continuous(
    context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear behavior policy function to generate synthetic continuous bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    random_state: int, default=None
        Controls the random seed in sampling parameters.

    Returns
    ---------
    expected_action_value: array-like, shape (n_rounds,)
        Expected continuous action values given context (:math:`x`).

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")

    random_ = check_random_state(random_state)
    coef_ = random_.normal(size=context.shape[1])
    bias = random_.uniform(size=1)
    return context @ coef_ + bias


# some functions to generate synthetic (evaluation) policies for continuous actions


def linear_synthetic_policy_continuous(context: np.ndarray) -> np.ndarray:
    """Linear synthtic policy for continuous actions.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    Returns
    ---------
    action_by_evaluation_policy: array-like, shape (n_rounds,)
        Continuous action values given by a synthetic (deterministic) evaluation policy, i.e., :math:`\\pi_e(x_t)`.

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")

    return context.mean(1)


def threshold_synthetic_policy_continuous(context: np.ndarray) -> np.ndarray:
    """Threshold synthtic policy for continuous actions.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    Returns
    ---------
    action_by_evaluation_policy: array-like, shape (n_rounds,)
        Continuous action values given by a synthetic (deterministic) evaluation policy, i.e., :math:`\\pi_e(x_t)`.

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")

    return 1.0 + np.sign(context.mean(1) - 1.5)


def sign_synthetic_policy_continuous(context: np.ndarray) -> np.ndarray:
    """Sign synthtic policy for continuous actions.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    Returns
    ---------
    action_by_evaluation_policy: array-like, shape (n_rounds,)
        Continuous action values given by a synthetic (deterministic) evaluation policy, i.e., :math:`\\pi_e(x_t)`.

    """
    if not isinstance(context, np.ndarray) or context.ndim != 2:
        raise ValueError("context must be 2-dimensional ndarray")

    return np.sin(context.mean(1))
