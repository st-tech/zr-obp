# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

from typing import Optional

import numpy as np
from numpy import log
from numpy import sqrt
from numpy import var
from scipy import stats
from sklearn.utils import check_scalar


def estimate_bias_in_ope(
    reward: np.ndarray,
    iw: np.ndarray,
    iw_hat: np.ndarray,
    q_hat: Optional[np.ndarray] = None,
) -> float:
    """Helper to estimate a bias in OPE.

    Parameters
    ----------
    reward: array-like, shape (n_rounds,)
        Rewards observed for each data in logged bandit data, i.e., :math:`r_t`.

    iw: array-like, shape (n_rounds,)
        Importance weight for each data in logged bandit data, i.e., :math:`w(x,a)=\\pi_e(a|x)/ \\pi_b(a|x)`.

    iw_hat: array-like, shape (n_rounds,)
        Importance weight (IW) modified by a hyparpareter. How IW is modified depends on the estimator as follows.
            - clipping: :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
            - switching: :math:`\\hat{w}(x,a) := w(x,a) \\cdot \\mathbb{I} \\{ w(x,a) < \\lambda \\}`
            - shrinkage: :math:`\\hat{w}(x,a) := (\\lambda w(x,a)) / (\\lambda + w^2(x,a))`
        where :math:`\\lambda` is a hyperparameter value.

    q_hat: array-like, shape (n_rounds,), default=None
        Estimated expected reward given context :math:`x_i` and action :math:`a_i`.

    Returns
    ----------
    estimated_bias: float
        Estimated the bias in OPE.
        This is based on the direct bias estimation stated on page 17 of Su et al.(2020).

    References
    ----------
    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """
    if q_hat is None:
        q_hat = np.zeros(reward.shape[0])
    estimated_bias_arr = (iw - iw_hat) * (reward - q_hat)
    estimated_bias = np.abs(estimated_bias_arr.mean())

    return estimated_bias


def estimate_high_probability_upper_bound_bias(
    reward: np.ndarray,
    iw: np.ndarray,
    iw_hat: np.ndarray,
    q_hat: Optional[np.ndarray] = None,
    delta: float = 0.05,
) -> float:
    """Helper to estimate a high probability upper bound of bias in OPE.

    Parameters
    ----------
    reward: array-like, shape (n_rounds,)
        Rewards observed for each data in logged bandit data, i.e., :math:`r_t`.

    iw: array-like, shape (n_rounds,)
        Importance weight for each data in logged bandit data, i.e., :math:`w(x,a)=\\pi_e(a|x)/ \\pi_b(a|x)`.

    iw_hat: array-like, shape (n_rounds,)
        Importance weight (IW) modified by a hyparpareter. How IW is modified depends on the estimator as follows.
            - clipping: :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
            - switching: :math:`\\hat{w}(x,a) := w(x,a) \\cdot \\mathbb{I} \\{ w(x,a) < \\lambda \\}`
            - shrinkage: :math:`\\hat{w}(x,a) := (\\lambda w(x,a)) / (\\lambda + w^2(x,a))`
        where :math:`\\lambda` and :math:`\\lambda` are hyperparameters.

    q_hat: array-like, shape (n_rounds,), default=None
        Estimated expected reward given context :math:`x_i` and action :math:`a_i`.

    delta: float, default=0.05
        A confidence delta to construct a high probability upper bound based on Bernstein inequality.

    Returns
    ----------
    bias_upper_bound: float
        Estimated (high probability) upper bound of the bias.
        This upper bound is based on the direct bias estimation
        stated on page 17 of Su et al.(2020).

    References
    ----------
    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """
    check_scalar(delta, "delta", (int, float), min_val=0.0, max_val=1.0)

    estimated_bias = estimate_bias_in_ope(
        reward=reward,
        iw=iw,
        iw_hat=iw_hat,
        q_hat=q_hat,
    )
    n = reward.shape[0]
    bias_upper_bound = estimated_bias
    bias_upper_bound += sqrt((2 * (iw**2).mean() * log(2 / delta)) / n)
    bias_upper_bound += (2 * iw.max() * log(2 / delta)) / (3 * n)

    return bias_upper_bound


def estimate_hoeffding_lower_bound(
    x: np.ndarray, x_max: Optional[float] = None, delta: float = 0.05
) -> float:
    """Estimate a high probability lower bound of mean of random variables by Hoeffding Inequality.

    Parameters
    ----------
    x: array-like, shape (n,)
        Size n of independent real-valued bounded random variables of interest.

    x_max: float, default=None.
        A maximum value of random variable `x`.
        If None, this is estimated from the given samples.

    delta: float, default=0.05
        A confidence delta to construct a high probability lower bound.

    Returns
    ----------
    lower_bound_estimate: float
        A high probability lower bound of mean of random variables `x` estimated by Hoeffding Inequality.
        See page 3 of Thomas et al.(2015) for details.

    References
    ----------
    Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh.
    "High Confidence Off-Policy Evaluation.", 2015.

    """
    if x_max is None:
        x_max = x.max()
    else:
        check_scalar(x_max, "x_max", (int, float), min_val=x.max())
    check_scalar(delta, "delta", (int, float), min_val=0.0, max_val=1.0)

    n = x.shape[0]
    ci = x_max * sqrt(log(1.0 / delta) / (2 * n))
    lower_bound_estimate = x.mean() - ci

    return lower_bound_estimate


def estimate_bernstein_lower_bound(
    x: np.ndarray, x_max: Optional[float], delta: float = 0.05
) -> float:
    """Estimate a high probability lower bound of mean of random variables by empirical Bernstein Inequality.

    Parameters
    ----------
    x: array-like, shape (n, )
        Size n of independent real-valued bounded random variables of interest.

    x_max: float, default=None.
        A maximum value of random variable `x`.
        If None, this is estimated from the given samples.

    delta: float, default=0.05
        A confidence delta to construct a high probability lower bound.

    Returns
    ----------
    lower_bound_estimate: float
        A high probability lower bound of mean of random variables `x` estimated by Hoeffding Inequality.
        See page 3 of Thomas et al.(2015) for details.

    References
    ----------
    Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh.
    "High Confidence Off-Policy Evaluation.", 2015.

    """
    if x_max is None:
        x_max = x.max()
    else:
        check_scalar(x_max, "x_max", (int, float), min_val=x.max())
    check_scalar(delta, "delta", (int, float), min_val=0.0, max_val=1.0)

    n = x.shape[0]
    ci1 = 7 * x_max * log(2.0 / delta) / (3 * (n - 1))
    ci2 = sqrt(2 * log(2.0 / delta) * var(x) / (n - 1))
    lower_bound_estimate = x.mean() - ci1 - ci2

    return lower_bound_estimate


def estimate_student_t_lower_bound(x: np.ndarray, delta: float = 0.05) -> float:
    """Estimate a high probability lower bound of mean of random variables based on Student t distribution.

    Parameters
    ----------
    x: array-like, shape (n, )
        Size n of independent real-valued bounded random variables of interest.

    delta: float, default=0.05
        A confidence delta to construct a high probability lower bound.

    Returns
    ----------
    lower_bound_estimate: float
        A high probability lower bound of mean of random variables `x` estimated based on Student t distribution.
        See Section 2.4 of Thomas et al.(2015) for details.

    References
    ----------
    Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh.
    "High Confidence Off-Policy Improvement.", 2015.

    """
    check_scalar(delta, "delta", (int, float), min_val=0.0, max_val=1.0)

    n = x.shape[0]
    ci = sqrt(var(x) / (n - 1))
    ci *= stats.t(n - 1).ppf(1.0 - delta)
    lower_bound_estimate = x.mean() - ci

    return lower_bound_estimate
