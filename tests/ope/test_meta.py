from typing import Dict, Optional
from dataclasses import dataclass

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from obp.types import BanditFeedback
from obp.ope import OffPolicyEvaluation, BaseOffPolicyEstimator


mock_policy_value = 0.5
mock_confidence_interval = {
    "mean": 0.5,
    "95.0% CI (lower)": 0.3,
    "95.0% CI (upper)": 0.7,
}


@dataclass
class DirectMethodMock(BaseOffPolicyEstimator):
    """Direct Method (DM) Mock."""

    estimator_name: str = "dm"

    def _estimate_round_rewards(
        self,
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        return 1

    def estimate_policy_value(
        self,
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        mock_policy_value: float
        """
        return mock_policy_value

    def estimate_interval(
        self,
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        mock_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.
        """
        return mock_confidence_interval


@dataclass
class InverseProbabilityWeightingMock(BaseOffPolicyEstimator):
    """Inverse Probability Weighting (IPW) Mock."""

    estimator_name: str = "ipw"
    eps: int = 0.1

    def _estimate_round_rewards(
        self,
        position: np.ndarray,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        return 1

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of an evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        mock_policy_value: float

        """
        return mock_policy_value + self.eps

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Positions of each round in the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities
            by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        alpha: float, default=0.05
            P-value.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        mock_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        return {k: v + self.eps for k, v in mock_confidence_interval.items()}


# define Mock instances
dm = DirectMethodMock()
ipw = InverseProbabilityWeightingMock(eps=0.01)
ipw2 = InverseProbabilityWeightingMock(eps=0.02)
ipw3 = InverseProbabilityWeightingMock(estimator_name="ipw3")


def test_meta_estimation_format(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the response format of OffPolicyEvaluation
    """
    # single ope estimator
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[dm]
    )
    assert ope_.estimate_policy_values(random_action_dist) == {
        "dm": mock_policy_value
    }, "OffPolicyEvaluation.estimate_policy_values ([DirectMethod]) returns a wrong value"
    assert ope_.estimate_intervals(random_action_dist) == {
        "dm": mock_confidence_interval
    }, "OffPolicyEvaluation.estimate_intervals ([DirectMethod]) returns a wrong value"
    with pytest.raises(AssertionError, match=r"action_dist must be 3-dimensional.*"):
        ope_.estimate_policy_values(
            random_action_dist[:, :, 0]
        ), "action_dist must be 3-dimensional when using OffPolicyEvaluation"
    # multiple ope estimators
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[dm, ipw]
    )
    assert ope_.estimate_policy_values(random_action_dist) == {
        "dm": mock_policy_value,
        "ipw": mock_policy_value + ipw.eps,
    }, "OffPolicyEvaluation.estimate_policy_values ([DirectMethod, IPW]) returns a wrong value"
    assert ope_.estimate_intervals(random_action_dist) == {
        "dm": mock_confidence_interval,
        "ipw": {k: v + ipw.eps for k, v in mock_confidence_interval.items()},
    }, "OffPolicyEvaluation.estimate_intervals ([DirectMethod]) returns a wrong value"


def test_meta_post_init_format(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the post init format of OffPolicyEvaluation
    """
    # __post_init__ saves the latter estimator when the same estimator name is used
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[ipw, ipw2]
    )
    assert ope_.ope_estimators_ == {"ipw": ipw2}, "__post_init__ returns a wrong value"
    # __post_init__ can handle the same estimator if the estimator names are different
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[ipw, ipw3]
    )
    assert ope_.ope_estimators_ == {
        "ipw": ipw,
        "ipw3": ipw3,
    }, "__post_init__ returns a wrong value"


def test_meta_create_estimator_inputs_format(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the _create_estimator_inputs format of OffPolicyEvaluation
    """
    # __post_init__ saves the latter estimator when the same estimator name is used
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[ipw]
    )
    inputs = ope_._create_estimator_inputs(
        action_dist=None, estimated_rewards_by_reg_model=None
    )
    assert set(inputs.keys()) == set(
        [
            "reward",
            "action",
            "pscore",
            "position",
            "action_dist",
            "estimated_rewards_by_reg_model",
        ]
    ), "Invalid response format of _create_estimator_inputs"


def test_meta_summarize_off_policy_estimates(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[ipw, ipw3]
    )
    value, interval = ope_.summarize_off_policy_estimates(random_action_dist)
    expected_value = pd.DataFrame(
        {
            "ipw": mock_policy_value + ipw.eps,
            "ipw3": mock_policy_value + ipw3.eps,
        },
        index=["estimated_policy_value"],
    ).T
    expected_interval = pd.DataFrame(
        {
            "ipw": {k: v + ipw.eps for k, v in mock_confidence_interval.items()},
            "ipw3": {k: v + ipw3.eps for k, v in mock_confidence_interval.items()},
        }
    ).T
    assert_frame_equal(value, expected_value), "Invalid summarization (policy value)"
    assert_frame_equal(interval, expected_interval), "Invalid summarization (interval)"


def test_meta_evaluate_performance_of_estimators(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    gt = 0.5
    # calculate relative-ee
    eval_metric_ope_dict = {
        "ipw": np.abs((mock_policy_value + ipw.eps - gt) / gt),
        "ipw3": np.abs((mock_policy_value + ipw3.eps - gt) / gt),
    }
    # check performance estimators
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[ipw, ipw3]
    )
    performance = ope_.evaluate_performance_of_estimators(
        ground_truth_policy_value=gt,
        action_dist=random_action_dist,
        metric="relative-ee",
    )
    for k, v in performance.items():
        assert k in eval_metric_ope_dict, "Invalid key of performance response"
        assert v == eval_metric_ope_dict[k], "Invalid value of performance response"
    # zero division error when using relative-ee
    with pytest.raises(ZeroDivisionError, match=r"float division by zero"):
        _ = ope_.evaluate_performance_of_estimators(
            ground_truth_policy_value=0.0,
            action_dist=random_action_dist,
            metric="relative-ee",
        )
    # check summarization
    performance_df = ope_.summarize_estimators_comparison(
        ground_truth_policy_value=gt,
        action_dist=random_action_dist,
        metric="relative-ee",
    )
    assert_frame_equal(
        performance_df, pd.DataFrame(eval_metric_ope_dict, index=["relative-ee"]).T
    ), "Invalid summarization (performance)"
