# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Functions for Generating Synthetic Logged Bandit Feedback."""
from typing import Optional

import numpy as np
from scipy.special import expit, softmax
from sklearn.utils import check_random_state, check_scalar


def logistic_reward_function(
    context: np.ndarray,
    action_context: Optional[np.ndarray] = None,
    n_actions: Optional[int] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function used to generate synthetic bandit datasets.

    Note
    --------
    This function generates a logistic mean reward function as follows:

    .. math::

        q(x, a) = \\mathbb{E}[r|x, a] = \\sigma (\\theta_a^{\\top} x +  \\eta_a^{\\top} \\phi(a)),

    where :math:`\\sigma(\\cdot)` is sigmoid function.
    :math:`\\theta_a` is a coefficient vector for the context vector and
    :math:`\\eta_a` is a coefficient vector for the action context vector where :math:`\\phi(\\cdot)` is an action representation function
    (one-hot encoding is a default one).
    Both :math:`\\theta_a` and :math:`\\eta_a` (:math:`\\forall a \\in \\mathcal{A}`) are sampled from the standard normal distribution.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context), default=None
        Vector representation for each action.
        When None is given, one-hot encoding is applied to represent each action by a vector.

    n_actions: int, default=None
        Number of actions. Must be set when `action_context` is None.

    random_state: int, default=None
        Controls the random seed in sampling coefficient vectors.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    assert (
        isinstance(context, np.ndarray) and context.ndim == 2
    ), "context must be 2-dimensional ndarray"
    if action_context is None:
        assert (
            n_actions is not None
        ), "n_actions must be set when action_context is None"
        check_scalar(n_actions, name="n_actions", target_type=(int), min_val=1)
        action_context = np.eye(n_actions)
    else:
        assert (
            isinstance(action_context, np.ndarray) and action_context.ndim == 2
        ), "action_context must be 2-dimensional ndarray"

    random_ = check_random_state(random_state)
    n_rounds, n_actions = context.shape[0], action_context.shape[0]
    dim_context, dim_action_context = context.shape[1], action_context.shape[1]
    # each arm has different coefficient vector
    coef_ = random_.normal(size=(n_actions, dim_context))
    action_coef_ = random_.normal(size=dim_action_context)

    logits = np.zeros((n_rounds, n_actions))
    for a in np.arange(action_context.shape[0]):
        logits[:, a] = context @ coef_[a] + action_context[a] @ action_coef_

    return expit(logits)


def linear_reward_function(
    context: np.ndarray,
    action_context: Optional[np.ndarray] = None,
    n_actions: Optional[int] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear mean reward function used to generate synthetic bandit datasets.

    Note
    --------
    This function generates a linear mean reward function as follows:

    .. math::

        q(x, a) = \\mathbb{E}[r|x, a] = \\theta_a^{\\top} x +  \\eta_a^{\\top} \\phi(a),

    where :math:`\\theta_a` is a coefficient vector for the context vector and
    :math:`\\eta_a` is a coefficient vector for the action context vector where :math:`\\phi(\\cdot)` is an action representation function
    (one-hot encoding is a default one).
    Both :math:`\\theta_a` and :math:`\\eta_a` (:math:`\\forall a \\in \\mathcal{A}`) are sampled from the standard normal distribution.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context), default=None
        Vector representation for each action.
        When None is given, one-hot encoding is applied to represent each action by a vector.

    n_actions: int, default=None
        Number of actions. Must be set when `action_context` is None.

    random_state: int, default=None
        Controls the random seed in sampling coefficient vectors.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    assert (
        isinstance(context, np.ndarray) and context.ndim == 2
    ), "context must be 2-dimensional ndarray"
    if action_context is None:
        assert (
            n_actions is not None
        ), "n_actions must be set when action_context is None"
        check_scalar(n_actions, name="n_actions", target_type=(int), min_val=1)
        action_context = np.eye(n_actions)
    else:
        assert (
            isinstance(action_context, np.ndarray) and action_context.ndim == 2
        ), "action_context must be 2-dimensional ndarray"

    random_ = check_random_state(random_state)
    n_rounds, n_actions = context.shape[0], action_context.shape[0]
    dim_context, dim_action_context = context.shape[1], action_context.shape[1]
    # each arm has different coefficient vector
    coef_ = random_.normal(size=(n_actions, dim_context))
    action_coef_ = random_.normal(size=dim_action_context)

    expected_reward = np.zeros((n_rounds, n_actions))
    for a in np.arange(n_actions):
        expected_reward[:, a] = context @ coef_[a] + action_context[a] @ action_coef_

    return expected_reward


def linear_policy_function(
    context: np.ndarray,
    action_context: Optional[np.ndarray] = None,
    n_actions: Optional[int] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear contextual policy function used to generate synthetic bandit datasets.

    Note
    --------
    This function generates a policy function as follows:

    .. math::

        \\pi (a|x) = \\mathbb{E}[a|x] =
        \\frac{e^{(\\theta_a^{\\top} x +  \\eta_a^{\\top} \\phi(a))}}
        {\\sum_{a^{\\prime} \\in \\mathcal{A}} e^{(\\theta_{a^{\\prime}}^{\\top} x +  \\eta_{a^{\\prime}}^{\\top} \\phi(a^{\\prime}))} },

    where :math:`\\theta_a` is a coefficient vector for the context vector and
    :math:`\\eta_a` is a coefficient vector for the action context vector where :math:`\\phi(\\cdot)` is an action representation function
    (one-hot encoding is a default one).
    Both :math:`\\theta_a` and :math:`\\eta_a` (:math:`\\forall a \\in \\mathcal{A}`) are sampled from the standard normal distribution.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context), default=None
        Vector representation for each action.
        When None is given, one-hot encoding is applied to represent each action by a vector.

    n_actions: int, default=None
        Number of actions. Must be set when `action_context` is None.

    random_state: int, default=None
        Controls the random seed in sampling coefficient vectors.

    Returns
    ---------
    policy_function: array-like, shape (n_rounds, n_actions)
        Action choice probabilities given context (:math:`x`), i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.

    """
    assert (
        isinstance(context, np.ndarray) and context.ndim == 2
    ), "context must be 2-dimensional ndarray"
    if action_context is None:
        assert (
            n_actions is not None
        ), "n_actions must be set when action_context is None"
        check_scalar(n_actions, name="n_actions", target_type=(int), min_val=1)
        action_context = np.eye(n_actions)
    else:
        assert (
            isinstance(action_context, np.ndarray) and action_context.ndim == 2
        ), "action_context must be 2-dimensional ndarray"

    random_ = check_random_state(random_state)
    n_rounds, n_actions = context.shape[0], action_context.shape[0]
    dim_context, dim_action_context = context.shape[1], action_context.shape[1]
    coef_ = random_.normal(size=(n_actions, dim_context))
    action_coef_ = random_.normal(size=dim_action_context)

    logits = np.zeros((n_rounds, n_actions))
    for a in np.arange(n_actions):
        logits[:, a] = context @ coef_[a] + action_context[a] @ action_coef_

    return softmax(logits, axis=1)
