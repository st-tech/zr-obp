# Copyright (c) ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

from inspect import isclass
from typing import Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import _deprecate_positional_args


def estimate_confidence_interval_by_bootstrap(samples: np.ndarray,
                                              alpha: float = 0.05,
                                              n_bootstrap_samples: int = 10000,
                                              random_state: Optional[int] = None) -> Dict[str, float]:
    """Estimate confidence interval by nonparametric bootstrap-like procedure.

    Parameters
    ----------
    samples: array
        Empirical observed samples to be used to estimate cumulative distribution function.

    alpha: float, default: 0.05
        P-value.

    n_bootstrap_samples: int, default: 10000
        Number of resampling performed in the bootstrap procedure.

    random_state: int, default: None
        Controls the random seed in bootstrap sampling.

    Returns
    ----------
    estimated_ci: Dict[str, float]
        Dictionary storing the estimated mean and upper-lower confidence bounds.
    """
    boot_samples = list()
    random_ = check_random_state(random_state)
    for _ in np.arange(n_bootstrap_samples):
        boot_samples.append(np.mean(random_.choice(samples, size=samples.shape[0])))
    lower = np.percentile(boot_samples, 100 * (alpha / 2))
    upper = np.percentile(boot_samples, 100 * (1. - alpha / 2))
    estimated_ci = dict(mean=np.mean(boot_samples), lower=lower, upper=upper)
    return estimated_ci


@_deprecate_positional_args
def check_is_fitted(estimator: BaseEstimator,
                    attributes=None,
                    *,
                    msg: str = None,
                    all_or_any=all) -> bool:
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
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]

    is_fitted = len(attrs) != 0
    return is_fitted


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Calculate sigmoid function."""
    return 1. / (1. + np.exp(-x))
