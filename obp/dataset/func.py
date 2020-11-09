# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Functions for Generating Synthetic Logged Bandit Feedback."""
from typing import Optional

import numpy as np
from scipy.special import expit, softmax
from sklearn.utils import check_random_state


def logistic_reward_function(
    context: np.ndarray, action_context: np.ndarray, random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    assert (
        isinstance(context, np.ndarray) and context.ndim == 2
    ), "context must be 2-dimensional ndarray"
    assert (
        isinstance(action_context, np.ndarray) and action_context.ndim == 2
    ), "action_context must be 2-dimensional ndarray"

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    # each arm has different coefficient vectors
    coef_ = random_.normal(size=(action_context.shape[0], context.shape[1]))
    action_coef_ = random_.normal(size=action_context.shape[1])
    for a in np.arange(action_context.shape[0]):
        logits[:, a] = context @ coef_[a] + action_context[a] @ action_coef_

    return expit(logits)


def linear_reward_function(
    context: np.ndarray, action_context: np.ndarray, random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear mean reward function for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    assert (
        isinstance(context, np.ndarray) and context.ndim == 2
    ), "context must be 2-dimensional ndarray"
    assert (
        isinstance(action_context, np.ndarray) and action_context.ndim == 2
    ), "action_context must be 2-dimensional ndarray"

    random_ = check_random_state(random_state)
    expected_reward = np.zeros((context.shape[0], action_context.shape[0]))
    # each arm has different coefficient vectors
    coef_ = random_.normal(size=(action_context.shape[0], context.shape[1]))
    action_coef_ = random_.normal(size=action_context.shape[1])
    for a in np.arange(action_context.shape[0]):
        expected_reward[:, a] = context @ coef_[a] + action_context[a] @ action_coef_

    return expected_reward


def linear_behavior_policy(
    context: np.ndarray, action_context: np.ndarray, random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear contextual behavior policy for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    behavior_policy: array-like, shape (n_rounds, n_actions)
        Action choice probabilities given context (:math:`x`), i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.

    """
    assert (
        isinstance(context, np.ndarray) and context.ndim == 2
    ), "context must be 2-dimensional ndarray"
    assert (
        isinstance(action_context, np.ndarray) and action_context.ndim == 2
    ), "action_context must be 2-dimensional ndarray"

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    coef_ = random_.normal(size=context.shape[1])
    action_coef_ = random_.normal(size=action_context.shape[1])
    for a in np.arange(action_context.shape[0]):
        logits[:, a] = context @ coef_ + action_context[a] @ action_coef_

    return softmax(logits, axis=1)
