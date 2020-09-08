# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Useful Tools."""
from inspect import isclass
from typing import Dict, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import _deprecate_positional_args


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 10000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate confidence interval by nonparametric bootstrap-like procedure.

    Parameters
    ----------
    samples: array-like
        Empirical observed samples to be used to estimate cumulative distribution function.

    alpha: float, default: 0.05
        P-value.

    n_bootstrap_samples: int, default: 10000
        Number of resampling performed in the bootstrap procedure.

    random_state: int, default: None
        Controls the random seed in bootstrap sampling.

    Returns
    ----------
    estimated_confidence_interval: Dict[str, float]
        Dictionary storing the estimated mean and upper-lower confidence bounds.

    """
    assert (0.0 < alpha < 1.0) and isinstance(
        alpha, float
    ), f"alpha must be a positive float, but {alpha} is given"
    assert (n_bootstrap_samples > 0) and isinstance(
        n_bootstrap_samples, int
    ), f"n_bootstrap_samples must be a positive integer, but {n_bootstrap_samples} is given"

    boot_samples = list()
    random_ = check_random_state(random_state)
    for _ in np.arange(n_bootstrap_samples):
        boot_samples.append(np.mean(random_.choice(samples, size=samples.shape[0])))
    lower_bound = np.percentile(boot_samples, 100 * (alpha / 2))
    upper_bound = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))
    return {
        "mean": np.mean(boot_samples),
        f"{100 * (1. - alpha)}% CI (lower)": lower_bound,
        f"{100 * (1. - alpha)}% CI (upper)": upper_bound,
    }


def convert_to_action_dist(n_actions: int, selected_actions: np.ndarray,) -> np.ndarray:
    """Convert selected actions (output of `run_bandit_simulation`) to distribution over actions.

    Parameters
    ----------
    n_actions: int
        Number of actions.

    selected_actions: array-like, shape (n_rounds, len_list)
            Sequence of actions selected by evaluation policy
            at each round in offline bandit simulation.

    Returns
    ----------
    action_dist: array-like shape (n_rounds, n_actions, len_list)
        Distribution over actions, i.e., probability of items being selected at each position (can be deterministic).

    """
    n_rounds, len_list = selected_actions.shape
    action_dist = np.zeros((n_rounds, n_actions, len_list))
    for pos in np.arange(len_list):
        selected_actions_ = selected_actions[:, pos]
        action_dist[
            np.arange(n_rounds), selected_actions_, pos * np.ones(n_rounds, int),
        ] = 1
    return action_dist


@_deprecate_positional_args
def check_is_fitted(
    estimator: BaseEstimator, attributes=None, *, msg: str = None, all_or_any=all
) -> bool:
    """Perform is_fitted validation for estimator.

    Note
    ----
    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.
    This utility is meant to be used internally by estimators themselves,
    typically in their own predict / transform methods.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``
        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    is_fitted: bool
        Whether the given estimator is fitted or not.

    References
    -------
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html

    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    is_fitted = len(attrs) != 0
    return is_fitted


def check_bandit_feedback_inputs(
    context: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    position: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
    action_context: Optional[np.ndarray] = None,
) -> Optional[AssertionError]:
    """Check inputs for bandit learning or simulation.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors in the given logged bandit feedback.

    action: array-like, shape (n_rounds,)
        Selected actions by behavior policy in the given logged bandit feedback.

    reward: array-like, shape (n_rounds,)
        Observed rewards in the given logged bandit feedback.

    position: array-like, shape (n_rounds,), default: None
        Positions of each round in the given logged bandit feedback.

    pscore: array-like, shape (n_rounds,), default: None
        Propensity scores, the probability of selecting each action by behavior policy,
        in the given logged bandit feedback.

    action_context: array-like, shape (n_actions, dim_action_context)
        Context vectors characterizing each action.

    """
    assert isinstance(context, np.ndarray), "context must be ndarray"
    assert context.ndim == 2, "context must be 2-dimensional"
    assert isinstance(action, np.ndarray), "action must be ndarray"
    assert action.ndim == 1, "action must be 1-dimensional"
    assert isinstance(reward, np.ndarray), "reward must be ndarray"
    assert reward.ndim == 1, "reward must be 1-dimensional"

    if pscore is not None:
        assert isinstance(pscore, np.ndarray), "pscore must be ndarray"
        assert pscore.ndim == 1, "pscore must be 1-dimensional"
        assert (
            context.shape[0] == action.shape[0] == reward.shape[0] == pscore.shape[0]
        ), "context, action, reward, and pscore must be the same size."
    if position is not None:
        assert isinstance(position, np.ndarray), "position must be ndarray"
        assert position.ndim == 1, "position must be 1-dimensional"
        assert (
            context.shape[0] == action.shape[0] == reward.shape[0] == position.shape[0]
        ), "context, action, reward, and position must be the same size."
    else:
        assert (
            context.shape[0] == action.shape[0] == reward.shape[0]
        ), "context, action, and reward must be the same size."
    if action_context is not None:
        assert isinstance(action_context, np.ndarray), "action_context must be ndarray"
        assert action_context.ndim == 2, "action_context must be 2-dimensional"
        assert (action.max() + 1) == action_context.shape[
            0
        ], "the number of action and the size of the first dimension of action_context must be same."


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate softmax function."""
    b = np.expand_dims(np.max(x, axis=1), 1)
    numerator = np.exp(x - b)
    denominator = np.expand_dims(np.sum(numerator, axis=1), 1)
    return numerator / denominator
