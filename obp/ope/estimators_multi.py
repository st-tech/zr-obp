# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators."""
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar

from ..utils import check_array
from ..utils import check_multi_loggers_ope_inputs
from ..utils import estimate_confidence_interval_by_bootstrap


@dataclass
class BaseMultiLoggersOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators for multiple loggers."""

    @abstractmethod
    def _estimate_round_rewards(self) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate the policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap."""
        raise NotImplementedError


@dataclass
class MultiLoggersNaiveInverseProbabilityWeighting(BaseMultiLoggersOffPolicyEstimator):
    """Multi-Loggers Inverse Probability Weighting (Multi-IPW) Estimator.

    Note
    -------
    This estimator is called Naive IPS in Agarwal et al.(2018) and Averaged IS in Kallus et al.(2021).

    Multi-IPW estimates the policy value of evaluation policy :math:`\\pi_e`
    using logged data collected by multiple logging/behavior policies as

    .. math::

        \\hat{V}_{\\mathrm{Multi-IPW}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [ w_{k_i}(x_i,a_i) r_i],

    where :math:`\\mathcal{D}_k=\\{(x_i,a_i,r_i)\\}_{i=1}^{n_k}` is logged bandit data with :math:`n_k` observations collected by
    the k-th behavior policy :math:`\\pi_k`. :math:`w_k(x,a):=\\pi_e (a|x)/\\pi_k (a|x)` is the importance weight given :math:`x` and :math:`a` computed for the k-th behavior policy.
    We can represent the whole logged bandit data as :math:`\\mathcal{D}=\\{(k_i,x_i,a_i,r_i)\\}_{i=1}^{n}` where :math:`k_i` is the index to indicate the logging/behavior policy that generates i-th data, i.e., :math:`\\pi_{k_i}`.
    Note that :math:`n := \\sum_{k=1}^K` is the total number of logged bandit data.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    When the clipping is applied, a large importance weight is clipped as :math:`\\hat{w}_k(x,a) := \\min \\{ \\lambda, w_k(x,a) \\}`, where :math:`\\lambda (>0)` is a hyperparameter to specify a maximum allowed importance weight.

    Multi-IPW applies the standard IPW to each stratum and takes the weighted average of the K datasets.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='multi_ipw'.
        Name of the estimator.

    References
    ------------
    Aman Agarwal, Soumya Basu, Tobias Schnabel, and Thorsten Joachims.
    "Effective Evaluation using Logged Bandit Feedback from Multiple Loggers.", 2018.

    Nathan Kallus, Yuta Saito, and Masatoshi Uehara.
    "Optimal Off-Policy Evaluation from Multiple Logging Policies.", 2021.

    """

    lambda_: float = np.inf
    use_estimated_pscore: bool = False
    estimator_name: str = "multi_ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)
        return reward * iw

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_k(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class MultiLoggersBalancedInverseProbabilityWeighting(
    BaseMultiLoggersOffPolicyEstimator
):
    """Multi-Loggers Balanced Inverse Probability Weighting (Multi-Bal-IPW) Estimator.

    Note
    -------
    This estimator is called Balanced IPS in Agarwal et al.(2018) and Standard IS in Kallus et al.(2021).
    Note that this estimator is different from `obp.ope.BalancedInverseProbabilityWeighting`, which is for the standard OPE setting.

    Multi-Bal-IPW estimates the policy value of evaluation policy :math:`\\pi_e`
    using logged data collected by multiple logging/behavior policies as

    .. math::

        \\hat{V}_{\\mathrm{Multi-Bal-IPW}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [ w_{avg}(x_i,a_i) r_i],

    where :math:`\\mathcal{D}_k=\\{(x_i,a_i,r_i)\\}_{i=1}^{n_k}` is logged bandit data with :math:`n_k` observations collected by
    the k-th behavior policy :math:`\\pi_k`.
    :math:`w_{avg}(x,a):=\\pi_e (a|x)/\\pi_{avg} (a|x)` is the importance weight given :math:`x` and :math:`a` computed for the *average* behavior policy, which is defined as :math:`\\pi_{avg}(a|x) := \\sum_{k=1}^K \\rho_k \\pi_k(a|x)`.
    We can represent the whole logged bandit data as :math:`\\mathcal{D}=\\{(k_i,x_i,a_i,r_i)\\}_{i=1}^{n}` where :math:`k_i` is the index to indicate the logging/behavior policy that generates i-th data, i.e., :math:`\\pi_{k_i}`.
    Note that :math:`n := \\sum_{k=1}^K` is the total number of logged bandit data, and :math:`\\rho_k := n_k / n` is the dataset proportions.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    When the clipping is applied, a large importance weight is clipped as :math:`\\hat{w}_{avg}(x,a) := \\min \\{ \\lambda, w_{avg}(x,a) \\}`, where :math:`\\lambda (>0)` is a hyperparameter to specify a maximum allowed importance weight.

    Multi-Bal-IPW applies the standard IPW based on the averaged logging/behavior policy :math:`\\pi_{avg}`.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='multi_bal_ipw'.
        Name of the estimator.

    References
    ------------
    Aman Agarwal, Soumya Basu, Tobias Schnabel, and Thorsten Joachims.
    "Effective Evaluation using Logged Bandit Feedback from Multiple Loggers.", 2018.

    Nathan Kallus, Yuta Saito, and Masatoshi Uehara.
    "Optimal Off-Policy Evaluation from Multiple Logging Policies.", 2021.

    """

    lambda_: float = np.inf
    use_estimated_pscore: bool = False
    estimator_name: str = "multi_bal_ipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore_avg: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore_avg: array-like, shape (n_rounds,)
            Action choice probabilities of the average logging/behavior policy, i.e., :math:`\\pi_{avg}(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore_avg` must be given.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        iw_avg = action_dist[np.arange(action.shape[0]), action, position] / pscore_avg
        # weight clipping
        if isinstance(iw_avg, np.ndarray):
            iw_avg = np.minimum(iw_avg, self.lambda_)
        return reward * iw_avg

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        pscore_avg: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore_avg: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        pscore_avg: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_{avg}(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore_avg` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore_avg: array-like, shape (n_rounds,), default=None
            Estimated average logging/behavior policy, i.e., :math:`\\hat{\\pi}_{avg}(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(
                array=estimated_pscore_avg, name="estimated_pscore_avg", expected_dim=1
            )
            pscore_ = estimated_pscore_avg
        else:
            check_array(array=pscore_avg, name="pscore_avg", expected_dim=1)
            pscore_ = pscore_avg

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore_avg=pscore_,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        pscore_avg: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore_avg: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        pscore_avg: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the average logging/behavior policy (propensity scores), i.e., :math:`\\pi_{avg}(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore_avg` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated logging/behavior policy, i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(
                array=estimated_pscore_avg, name="estimated_pscore_avg", expected_dim=1
            )
            pscore_ = estimated_pscore_avg
        else:
            check_array(array=pscore_avg, name="pscore_avg", expected_dim=1)
            pscore_ = pscore_avg

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class MultiLoggersWeightedInverseProbabilityWeighting(
    MultiLoggersNaiveInverseProbabilityWeighting
):
    """Multi-Loggers Weighted Inverse Probability Weighting (Multi-Weighted-IPW) Estimator.

    Note
    -------
    This estimator is called Weighted IPS in Agarwal et al.(2018) and Precision Weighted IS in Kallus et al.(2021).

    Multi-Weighted-IPW estimates the policy value of evaluation policy :math:`\\pi_e`
    using logged data collected by multiple logging/behavior policies as

    .. math::

        \\hat{V}_{\\mathrm{Multi-Weighted-IPW}} (\\pi_e; \\mathcal{D})
        := \\sum_{k=1}^K \\M^*_k \\mathbb{E}_{n_k} [ w_k(x_i,a_i) r_i],

    where :math:`\\mathcal{D}_k=\\{(x_i,a_i,r_i)\\}_{i=1}^{n_k}` is logged bandit data with :math:`n_k` observations collected by
    the k-th behavior policy :math:`\\pi_k`. :math:`w_k(x,a):=\\pi_e (a|x)/\\pi_k (a|x)` is the importance weight given :math:`x` and :math:`a` computed for the k-th behavior policy.
    We can represent the whole logged bandit data as :math:`\\mathcal{D}=\\{(k_i,x_i,a_i,r_i)\\}_{i=1}^{n}` where :math:`k_i` is the index to indicate the logging/behavior policy that generates i-th data, i.e., :math:`\\pi_{k_i}`.
    Note that :math:`n := \\sum_{k=1}^K` is the total number of logged bandit data, and :math:`\\rho_k := n_k / n` is the dataset proportions.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    When the clipping is applied, a large importance weight is clipped as :math:`\\hat{w}_k(x,a) := \\min \\{ \\lambda, w_k(x,a) \\}`, where :math:`\\lambda (>0)` is a hyperparameter to specify a maximum allowed importance weight.

    Multi-Weighted-IPW prioritizes the strata generated by the logging/behavior policies similar to the evaluation policy.
    The weight for the k-th logging/behavior policy :math:`\\M^*_k` is defined based on
    the divergence between the evaluation policy :math:`\\pi_e` and :math:`\\pi_k`.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='multi_weighted_ipw'.
        Name of the estimator.

    References
    ------------
    Aman Agarwal, Soumya Basu, Tobias Schnabel, and Thorsten Joachims.
    "Effective Evaluation using Logged Bandit Feedback from Multiple Loggers.", 2018.

    Nathan Kallus, Yuta Saito, and Masatoshi Uehara.
    "Optimal Off-Policy Evaluation from Multiple Logging Policies.", 2021.

    """

    estimator_name: str = "multi_weighted_ipw"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        stratum_idx: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        stratum_idx: array-like, shape (n_rounds,)
            Indices to differentiate the logging/behavior policy that generate each data, i.e., :math:`k`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        n = action.shape[0]
        iw = action_dist[np.arange(n), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)

        unique_stratum_idx, n_data_strata = np.unique(stratum_idx, return_counts=True)
        var_k = np.zeros(unique_stratum_idx.shape[0])
        for k in unique_stratum_idx:
            idx_ = stratum_idx == k
            var_k[k] = np.var(reward[idx_] * iw[idx_])
        weight_k = n / (var_k * np.sum(n_data_strata / var_k))

        return reward * iw * weight_k[stratum_idx]

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        stratum_idx: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        stratum_idx: array-like, shape (n_rounds,)
            Indices to differentiate the logging/behavior policy that generate each data, i.e., :math:`k`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_k(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=stratum_idx, name="stratum_idx", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            stratum_idx=stratum_idx,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            stratum_idx=stratum_idx,
            action_dist=action_dist,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        stratum_idx: np.ndarray,
        action_dist: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        stratum_idx: array-like, shape (n_rounds,)
            Indices to differentiate the logging/behavior policy that generate each data, i.e., :math:`k_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=stratum_idx, name="stratum_idx", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            stratum_idx=stratum_idx,
            pscore=pscore_,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            stratum_idx=stratum_idx,
            pscore=pscore_,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class MultiLoggersNaiveDoublyRobust(BaseMultiLoggersOffPolicyEstimator):
    """Multi-Loggers Naive Doubly Robust (Multi-Naive-DR) Estimator.

    Note
    -------
    This estimator is called Average DR in Kallus et al.(2021).

    Multi-Naive-DR estimates the policy value of evaluation policy :math:`\\pi_e`
    using logged data collected by multiple logging/behavior policies as

    .. math::

        \\hat{V}_{\\mathrm{Multi-Naive-DR}} (\\pi_e; \\mathcal{D}, \\hat{q})
        := \\mathbb{E}_{n} [\\hat{q}(x_i,\\pi_e) + w_{k_i}(x_i,a_i) (r_i - \\hat{q}(x_i,a_i))],

    where :math:`\\mathcal{D}_k=\\{(x_i,a_i,r_i)\\}_{i=1}^{n_k}` is logged bandit data with :math:`n_k` observations collected by
    the k-th behavior policy :math:`\\pi_k`. :math:`w_k(x,a):=\\pi_e (a|x)/\\pi_k (a|x)` is the importance weight given :math:`x` and :math:`a` computed for the k-th behavior policy.
    We can represent the whole logged bandit data as :math:`\\mathcal{D}=\\{(k_i,x_i,a_i,r_i)\\}_{i=1}^{n}` where :math:`k_i` is the index to indicate the logging/behavior policy that generates i-th data, i.e., :math:`\\pi_{k_i}`.
    Note that :math:`n := \\sum_{k=1}^K` is the total number of logged bandit data.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is the estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_i,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    When the clipping is applied, a large importance weight is clipped as :math:`\\hat{w}_k(x,a) := \\min \\{ \\lambda, w_k(x,a) \\}`, where :math:`\\lambda (>0)` is a hyperparameter to specify a maximum allowed importance weight.

    Multi-Naive-DR applies the standard DR to each stratum and takes the weighted average of the K datasets.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='multi_dr'.
        Name of the estimator.

    References
    ------------
    Aman Agarwal, Soumya Basu, Tobias Schnabel, and Thorsten Joachims.
    "Effective Evaluation using Logged Bandit Feedback from Multiple Loggers.", 2018.

    Nathan Kallus, Yuta Saito, and Masatoshi Uehara.
    "Optimal Off-Policy Evaluation from Multiple Logging Policies.", 2021.

    """

    lambda_: float = np.inf
    use_estimated_pscore: bool = False
    estimator_name: str = "multi_dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)

        n = action.shape[0]
        q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n), :, position]
        q_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action, position]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += iw * (reward - q_hat_factual)

        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_k(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class MultiLoggersBalancedDoublyRobust(BaseMultiLoggersOffPolicyEstimator):
    """Multi-Loggers Balanced DoublyRobust (Multi-Bal-DR) Estimator.

    Note
    -------
    This estimator is called DR in Kallus et al.(2021).

    Multi-Bal-DR estimates the policy value of evaluation policy :math:`\\pi_e`
    using logged data collected by multiple logging/behavior policies as

    .. math::

        \\hat{V}_{\\mathrm{Multi-Bal-DR}} (\\pi_e; \\mathcal{D}, \\hat{q})
        := \\mathbb{E}_{n} [ \\hat{q}(x_i,\\pi_e) w_{avg}(x_i,a_i) (r_i - \\hat{q}(x_i,a_i))],

    where :math:`\\mathcal{D}_k=\\{(x_i,a_i,r_i)\\}_{i=1}^{n_k}` is logged bandit data with :math:`n_k` observations collected by
    the k-th behavior policy :math:`\\pi_k`.
    :math:`w_{avg}(x,a):=\\pi_e (a|x)/\\pi_{avg} (a|x)` is the importance weight given :math:`x` and :math:`a` computed for the *average* behavior policy, which is defined as :math:`\\pi_{avg}(a|x) := \\sum_{k=1}^K \\rho_k \\pi_k(a|x)`.
    We can represent the whole logged bandit data as :math:`\\mathcal{D}=\\{(k_i,x_i,a_i,r_i)\\}_{i=1}^{n}` where :math:`k_i` is the index to indicate the logging/behavior policy that generates i-th data, i.e., :math:`\\pi_{k_i}`.
    Note that :math:`n := \\sum_{k=1}^K` is the total number of logged bandit data, and :math:`\\rho_k := n_k / n` is the dataset proportions.
    :math:`\\hat{q} (x,a)` is the estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_i,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    When the clipping is applied, a large importance weight is clipped as :math:`\\hat{w}_{avg}(x,a) := \\min \\{ \\lambda, w_{avg}(x,a) \\}`, where :math:`\\lambda (>0)` is a hyperparameter to specify a maximum allowed importance weight.

    Multi-Bal-DR applies the standard DR based on the averaged logging/behavior policy :math:`\\pi_{avg}`.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='multi_bal_dr'.
        Name of the estimator.

    References
    ------------
    Aman Agarwal, Soumya Basu, Tobias Schnabel, and Thorsten Joachims.
    "Effective Evaluation using Logged Bandit Feedback from Multiple Loggers.", 2018.

    Nathan Kallus, Yuta Saito, and Masatoshi Uehara.
    "Optimal Off-Policy Evaluation from Multiple Logging Policies.", 2021.

    """

    lambda_: float = np.inf
    use_estimated_pscore: bool = False
    estimator_name: str = "multi_bal_dr"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(
            self.lambda_,
            name="lambda_",
            target_type=(int, float),
            min_val=0.0,
        )
        if self.lambda_ != self.lambda_:
            raise ValueError("`lambda_` must not be nan")
        if not isinstance(self.use_estimated_pscore, bool):
            raise TypeError(
                f"`use_estimated_pscore` must be a bool, but {type(self.use_estimated_pscore)} is given"
            )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore_avg: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore_avg: array-like, shape (n_rounds,)
            Action choice probabilities of the average logging/behavior policy, i.e., :math:`\\pi_{avg}(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore_avg` must be given.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        iw_avg = action_dist[np.arange(action.shape[0]), action, position] / pscore_avg
        # weight clipping
        if isinstance(iw_avg, np.ndarray):
            iw_avg = np.minimum(iw_avg, self.lambda_)

        n = action.shape[0]
        q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n), :, position]
        q_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action, position]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )
        estimated_rewards += iw_avg * (reward - q_hat_factual)

        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore_avg: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore_avg: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore_avg: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_{avg}(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore_avg` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore_avg: array-like, shape (n_rounds,), default=None
            Estimated average logging/behavior policy, i.e., :math:`\\hat{\\pi}_{avg}(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(
                array=estimated_pscore_avg, name="estimated_pscore_avg", expected_dim=1
            )
            pscore_ = estimated_pscore_avg
        else:
            check_array(array=pscore_avg, name="pscore_avg", expected_dim=1)
            pscore_ = pscore_avg

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore_avg=pscore_,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore_avg: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore_avg: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore_avg: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the average logging/behavior policy (propensity scores), i.e., :math:`\\pi_{avg}(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore_avg` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated logging/behavior policy, i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(
                array=estimated_pscore_avg, name="estimated_pscore_avg", expected_dim=1
            )
            pscore_ = estimated_pscore_avg
        else:
            check_array(array=pscore_avg, name="pscore_avg", expected_dim=1)
            pscore_ = pscore_avg

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class MultiLoggersWeightedDoublyRobust(MultiLoggersNaiveDoublyRobust):
    """Multi-Loggers Weighted Doubly Robust (Multi-Weighted-DR) Estimator.

    Note
    -------
    This estimator is called Precision Weighted DR in Kallus et al.(2021).

    Multi-Weighted-DR estimates the policy value of evaluation policy :math:`\\pi_e`
    using logged data collected by multiple logging/behavior policies as

    .. math::

        \\hat{V}_{\\mathrm{Multi-Weighted-DR}} (\\pi_e; \\mathcal{D}, \\hat{q})
        := \\sum_{k=1}^K \\M^{*}_k \\mathbb{E}_{n_k} [\\hat{q}(x_i,\\pi_e) + w_k(x_i,a_i) (r_i - \\hat{q}(x_i,a_i))],

    where :math:`\\mathcal{D}_k=\\{(x_i,a_i,r_i)\\}_{i=1}^{n_k}` is logged bandit data with :math:`n_k` observations collected by
    the k-th behavior policy :math:`\\pi_k`. :math:`w_k(x,a):=\\pi_e (a|x)/\\pi_k (a|x)` is the importance weight given :math:`x` and :math:`a` computed for the k-th behavior policy.
    We can represent the whole logged bandit data as :math:`\\mathcal{D}=\\{(k_i,x_i,a_i,r_i)\\}_{i=1}^{n}` where :math:`k_i` is the index to indicate the logging/behavior policy that generates i-th data, i.e., :math:`\\pi_{k_i}`.
    Note that :math:`n := \\sum_{k=1}^K` is the total number of logged bandit data, and :math:`\\rho_k := n_k / n` is the dataset proportions.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.
    :math:`\\hat{q} (x,a)` is the estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\hat{q} (x_i,\\pi):= \\mathbb{E}_{a \\sim \\pi(a|x)}[\\hat{q}(x,a)]` is the expectation of the estimated reward function over :math:`\\pi`.
    When the clipping is applied, a large importance weight is clipped as :math:`\\hat{w}_k(x,a) := \\min \\{ \\lambda, w_k(x,a) \\}`, where :math:`\\lambda (>0)` is a hyperparameter to specify a maximum allowed importance weight.

    Multi-Weighted-DR prioritizes the strata generated by the logging/behavior policies similar to the evaluation policy.
    The weight for the k-th logging/behavior policy :math:`\\M^*_k` is defined based on
    the divergence between the evaluation policy :math:`\\pi_e` and :math:`\\pi_k`.

    Parameters
    ------------
    lambda_: float, default=np.inf
        A maximum possible value of the importance weight.
        When a positive finite value is given, importance weights larger than `lambda_` will be clipped.

    use_estimated_pscore: bool, default=False.
        If True, `estimated_pscore` is used, otherwise, `pscore` (the true propensity scores) is used.

    estimator_name: str, default='multi_weighted_dr'.
        Name of the estimator.

    References
    ------------
    Aman Agarwal, Soumya Basu, Tobias Schnabel, and Thorsten Joachims.
    "Effective Evaluation using Logged Bandit Feedback from Multiple Loggers.", 2018.

    Nathan Kallus, Yuta Saito, and Masatoshi Uehara.
    "Optimal Off-Policy Evaluation from Multiple Logging Policies.", 2021.

    """

    estimator_name: str = "multi_weighted_dr"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        stratum_idx: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        stratum_idx: array-like, shape (n_rounds,)
            Indices to differentiate the logging/behavior policy that generate each data, i.e., :math:`k`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
        # weight clipping
        if isinstance(iw, np.ndarray):
            iw = np.minimum(iw, self.lambda_)

        n = action.shape[0]
        q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n), :, position]
        q_hat_factual = estimated_rewards_by_reg_model[np.arange(n), action, position]
        pi_e_at_position = action_dist[np.arange(n), :, position]
        estimated_rewards = np.average(
            q_hat_at_position,
            weights=pi_e_at_position,
            axis=1,
        )

        unique_stratum_idx, n_data_strata = np.unique(stratum_idx, return_counts=True)
        var_k = np.zeros(unique_stratum_idx.shape[0])
        for k in unique_stratum_idx:
            idx_ = stratum_idx == k
            var_k[k] = np.var(
                estimated_rewards[idx_]
                + iw[idx_] * (reward[idx_] - q_hat_factual[idx_])
            )
        weight_k = n / (var_k * np.sum(n_data_strata / var_k))
        estimated_rewards += iw * (reward - q_hat_factual) * weight_k[stratum_idx]

        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        stratum_idx: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        stratum_idx: array-like, shape (n_rounds,)
            Indices to differentiate the logging/behavior policy that generate each data, i.e., :math:`k`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_k(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=stratum_idx, name="stratum_idx", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            stratum_idx=stratum_idx,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        return self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            pscore=pscore_,
            stratum_idx=stratum_idx,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        stratum_idx: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_pscore: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        stratum_idx: array-like, shape (n_rounds,)
            Indices to differentiate the logging/behavior policy that generate each data, i.e., :math:`k_i`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_k(a_i|x_i)`.
            If `use_estimated_pscore` is False, `pscore` must be given.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            If `self.use_estimated_pscore` is True, `estimated_pscore` must be given.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=3,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=stratum_idx, name="stratum_idx", expected_dim=1)
        if self.use_estimated_pscore:
            check_array(array=estimated_pscore, name="estimated_pscore", expected_dim=1)
            pscore_ = estimated_pscore
        else:
            check_array(array=pscore, name="pscore", expected_dim=1)
            pscore_ = pscore

        check_multi_loggers_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
            stratum_idx=stratum_idx,
            pscore=pscore_,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action=action,
            position=position,
            stratum_idx=stratum_idx,
            pscore=pscore_,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
