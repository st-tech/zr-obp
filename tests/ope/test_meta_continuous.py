from typing import Dict, Optional
from dataclasses import dataclass
import itertools
from copy import deepcopy

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from obp.types import BanditFeedback
from obp.ope import (
    ContinuousOffPolicyEvaluation,
    BaseContinuousOffPolicyEstimator,
)
from obp.utils import check_confidence_interval_arguments


mock_policy_value = 0.5
mock_confidence_interval = {
    "mean": 0.5,
    "95.0% CI (lower)": 0.3,
    "95.0% CI (upper)": 0.7,
}


@dataclass
class KernelizedInverseProbabilityWeightingMock(BaseContinuousOffPolicyEstimator):
    """Mock Kernelized Inverse Probability Weighting."""

    eps: float = 0.1
    estimator_name: str = "ipw"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return np.ones_like(reward)

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Returns
        ----------
        mock_policy_value: float

        """
        return mock_policy_value + self.eps

    def estimate_interval(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Returns
        ----------
        mock_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_confidence_interval_arguments(
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        return {k: v + self.eps for k, v in mock_confidence_interval.items()}


@dataclass
class KernelizedDoublyRobustMock(BaseContinuousOffPolicyEstimator):
    """Mock Kernelized Doubly Robust."""

    estimator_name: str = "kernelized_dr"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return np.ones_like(reward)

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate policy value of an evaluation policy.

        Returns
        ----------
        mock_policy_value: float

        """
        return mock_policy_value

    def estimate_interval(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate confidence interval of policy value by nonparametric bootstrap procedure.

        Returns
        ----------
        mock_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_confidence_interval_arguments(
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        return {k: v for k, v in mock_confidence_interval.items()}


# define Mock instances
ipw = KernelizedInverseProbabilityWeightingMock()
ipw2 = KernelizedInverseProbabilityWeightingMock(eps=0.02)
ipw3 = KernelizedInverseProbabilityWeightingMock(estimator_name="ipw3")
dr = KernelizedDoublyRobustMock(estimator_name="dr")


def test_meta_post_init(synthetic_continuous_bandit_feedback: BanditFeedback) -> None:
    """
    Test the __post_init__ function
    """
    # __post_init__ saves the latter estimator when the same estimator name is used
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[ipw, ipw2]
    )
    assert ope_.ope_estimators_ == {"ipw": ipw2}, "__post_init__ returns a wrong value"
    # __post_init__ can handle the same estimator if the estimator names are different
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[ipw, ipw3]
    )
    assert ope_.ope_estimators_ == {
        "ipw": ipw,
        "ipw3": ipw3,
    }, "__post_init__ returns a wrong value"
    # __post__init__ raises RuntimeError when necessary_keys are not included in the bandit_feedback
    necessary_keys = ["action_by_behavior_policy", "reward", "pscore"]
    for i in range(len(necessary_keys)):
        for deleted_keys in itertools.combinations(necessary_keys, i + 1):
            invalid_bandit_feedback_dict = {key: "_" for key in necessary_keys}
            # delete
            for k in deleted_keys:
                del invalid_bandit_feedback_dict[k]
            with pytest.raises(RuntimeError, match=r"Missing key*"):
                _ = ContinuousOffPolicyEvaluation(
                    bandit_feedback=invalid_bandit_feedback_dict, ope_estimators=[ipw]
                )


# action_by_evaluation_policy, estimated_rewards_by_reg_model, description
invalid_input_of_create_estimator_inputs = [
    (
        np.zeros(5),  #
        np.zeros(4),  #
        "estimated_rewards_by_reg_model.shape and action_by_evaluation_policy.shape must be the same",
    ),
    (
        np.zeros(5),
        {"dr": np.zeros(4)},
        r"estimated_rewards_by_reg_model\[dr\].shape and action_by_evaluation_policy.shape must be the same",
    ),
    (
        np.zeros(5),
        {"dr": None},
        r"estimated_rewards_by_reg_model\[dr\] must be ndarray",
    ),
    (
        np.zeros((2, 3)),
        None,
        "action_by_evaluation_policy must be 1-dimensional ndarray",
    ),
    ("3", None, "action_by_evaluation_policy must be 1-dimensional ndarray"),
    (None, None, "action_by_evaluation_policy must be 1-dimensional ndarray"),
]

valid_input_of_create_estimator_inputs = [
    (
        np.zeros(5),
        np.zeros(5),
        "same shape",
    ),
    (
        np.zeros(5),
        {"dr": np.zeros(5)},
        "same shape",
    ),
    (np.zeros(5), None, "estimated_rewards_by_reg_model is None"),
]


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, description",
    invalid_input_of_create_estimator_inputs,
)
def test_meta_create_estimator_inputs_using_invalid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    description: str,
    synthetic_continuous_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the _create_estimator_inputs using valid data
    """
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[ipw]
    )
    # raise ValueError when the shape of two arrays are different
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_._create_estimator_inputs(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    # _create_estimator_inputs function is called in the following functions
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.estimate_policy_values(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.estimate_intervals(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.summarize_off_policy_estimates(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.evaluate_performance_of_estimators(
            ground_truth_policy_value=0.1,
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.summarize_estimators_comparison(
            ground_truth_policy_value=0.1,
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, description",
    valid_input_of_create_estimator_inputs,
)
def test_meta_create_estimator_inputs_using_valid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    description: str,
    synthetic_continuous_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the _create_estimator_inputs using invalid data
    """
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[ipw]
    )
    estimator_inputs = ope_._create_estimator_inputs(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    assert set(estimator_inputs.keys()) == set(["ipw"])
    assert set(estimator_inputs["ipw"].keys()) == set(
        [
            "reward",
            "action_by_behavior_policy",
            "pscore",
            "action_by_evaluation_policy",
            "estimated_rewards_by_reg_model",
        ]
    ), f"Invalid response of _create_estimator_inputs (test case: {description})"
    # _create_estimator_inputs function is called in the following functions
    _ = ope_.estimate_policy_values(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    _ = ope_.estimate_intervals(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    _ = ope_.summarize_off_policy_estimates(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    _ = ope_.evaluate_performance_of_estimators(
        ground_truth_policy_value=0.1,
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    _ = ope_.summarize_estimators_comparison(
        ground_truth_policy_value=0.1,
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, description",
    valid_input_of_create_estimator_inputs,
)
def test_meta_estimate_policy_values_using_valid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    description: str,
    synthetic_continuous_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of estimate_policy_values using valid data
    """
    # single ope estimator
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[dr]
    )
    assert ope_.estimate_policy_values(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    ) == {
        "dr": mock_policy_value
    }, "OffPolicyEvaluation.estimate_policy_values ([DoublyRobust]) returns a wrong value"
    # multiple ope estimators
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[dr, ipw]
    )
    assert ope_.estimate_policy_values(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    ) == {
        "dr": mock_policy_value,
        "ipw": mock_policy_value + ipw.eps,
    }, "OffPolicyEvaluation.estimate_policy_values ([DoublyRobust, IPW]) returns a wrong value"


# alpha, n_bootstrap_samples, random_state, description
invalid_input_of_estimate_intervals = [
    (0.05, 100, "s", "random_state must be an integer"),
    (0.05, -1, 1, "n_bootstrap_samples must be a positive integer"),
    (0.05, "s", 1, "n_bootstrap_samples must be a positive integer"),
    (0.0, 1, 1, "alpha must be a positive float (< 1)"),
    (1.0, 1, 1, "alpha must be a positive float (< 1)"),
    ("0", 1, 1, "alpha must be a positive float (< 1)"),
]

valid_input_of_estimate_intervals = [
    (0.05, 100, 1, "random_state is 1"),
    (0.05, 1, 1, "n_bootstrap_samples is 1"),
]


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    invalid_input_of_estimate_intervals,
)
def test_meta_estimate_intervals_using_invalid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    description_1: str,
    alpha,
    n_bootstrap_samples,
    random_state,
    description_2: str,
    synthetic_continuous_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of estimate_intervals using invalid data
    """
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[dr]
    )
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = ope_.estimate_intervals(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
    # estimate_intervals function is called in summarize_off_policy_estimates
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = ope_.summarize_off_policy_estimates(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    valid_input_of_estimate_intervals,
)
def test_meta_estimate_intervals_using_valid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    description_1: str,
    alpha: float,
    n_bootstrap_samples: int,
    random_state: int,
    description_2: str,
    synthetic_continuous_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of estimate_intervals using valid data
    """
    # single ope estimator
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[dr]
    )
    assert ope_.estimate_intervals(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    ) == {
        "dr": mock_confidence_interval
    }, "OffPolicyEvaluation.estimate_intervals ([DoublyRobust]) returns a wrong value"
    # multiple ope estimators
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[dr, ipw]
    )
    assert ope_.estimate_intervals(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    ) == {
        "dr": mock_confidence_interval,
        "ipw": {k: v + ipw.eps for k, v in mock_confidence_interval.items()},
    }, "OffPolicyEvaluation.estimate_intervals ([DoublyRobust, IPW]) returns a wrong value"


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    valid_input_of_estimate_intervals,
)
def test_meta_summarize_off_policy_estimates(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    description_1: str,
    alpha: float,
    n_bootstrap_samples: int,
    random_state: int,
    description_2: str,
    synthetic_continuous_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of summarize_off_policy_estimates using valid data
    """
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[ipw, ipw3]
    )
    value, interval = ope_.summarize_off_policy_estimates(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )
    expected_value = pd.DataFrame(
        {
            "ipw": mock_policy_value + ipw.eps,
            "ipw3": mock_policy_value + ipw3.eps,
        },
        index=["estimated_policy_value"],
    ).T
    expected_value["relative_estimated_policy_value"] = (
        expected_value["estimated_policy_value"]
        / synthetic_continuous_bandit_feedback["reward"].mean()
    )
    expected_interval = pd.DataFrame(
        {
            "ipw": {k: v + ipw.eps for k, v in mock_confidence_interval.items()},
            "ipw3": {k: v + ipw3.eps for k, v in mock_confidence_interval.items()},
        }
    ).T
    assert_frame_equal(value, expected_value), "Invalid summarization (policy value)"
    assert_frame_equal(interval, expected_interval), "Invalid summarization (interval)"
    # check relative estimated policy value when the average of bandit_feedback["reward"] is zero
    zero_reward_bandit_feedback = deepcopy(synthetic_continuous_bandit_feedback)
    zero_reward_bandit_feedback["reward"] = np.zeros(
        zero_reward_bandit_feedback["reward"].shape[0]
    )
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=zero_reward_bandit_feedback, ope_estimators=[ipw, ipw3]
    )
    value, _ = ope_.summarize_off_policy_estimates(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )
    expected_value = pd.DataFrame(
        {
            "ipw": mock_policy_value + ipw.eps,
            "ipw3": mock_policy_value + ipw3.eps,
        },
        index=["estimated_policy_value"],
    ).T
    expected_value["relative_estimated_policy_value"] = np.nan
    assert_frame_equal(value, expected_value), "Invalid summarization (policy value)"


invalid_input_of_evaluation_performance_of_estimators = [
    ("foo", 0.3, "metric must be either 'relative-ee' or 'se'"),
    ("se", 1, "ground_truth_policy_value must be a float"),
    ("se", "a", "ground_truth_policy_value must be a float"),
    (
        "relative-ee",
        0.0,
        "ground_truth_policy_value must be non-zero when metric is relative-ee",
    ),
]

valid_input_of_evaluation_performance_of_estimators = [
    ("se", 0.0, "metric is se and ground_truth_policy_value is 0.0"),
    ("relative-ee", 1.0, "metric is relative-ee and ground_truth_policy_value is 1.0"),
]


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "metric, ground_truth_policy_value, description_2",
    invalid_input_of_evaluation_performance_of_estimators,
)
def test_meta_evaluate_performance_of_estimators_using_invalid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    description_1: str,
    metric,
    ground_truth_policy_value,
    description_2: str,
    synthetic_continuous_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of evaluate_performance_of_estimators using invalid data
    """
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[dr]
    )
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = ope_.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_by_evaluation_policy=action_by_evaluation_policy,
            metric=metric,
        )
    # estimate_intervals function is called in summarize_off_policy_estimates
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = ope_.summarize_estimators_comparison(
            ground_truth_policy_value=ground_truth_policy_value,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_by_evaluation_policy=action_by_evaluation_policy,
            metric=metric,
        )


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "metric, ground_truth_policy_value, description_2",
    valid_input_of_evaluation_performance_of_estimators,
)
def test_meta_evaluate_performance_of_estimators_using_valid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    description_1: str,
    metric,
    ground_truth_policy_value,
    description_2: str,
    synthetic_continuous_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of evaluate_performance_of_estimators using valid data
    """
    if metric == "relative-ee":
        # calculate relative-ee
        eval_metric_ope_dict = {
            "ipw": np.abs(
                (mock_policy_value + ipw.eps - ground_truth_policy_value)
                / ground_truth_policy_value
            ),
            "ipw3": np.abs(
                (mock_policy_value + ipw3.eps - ground_truth_policy_value)
                / ground_truth_policy_value
            ),
        }
    else:
        # calculate se
        eval_metric_ope_dict = {
            "ipw": (mock_policy_value + ipw.eps - ground_truth_policy_value) ** 2,
            "ipw3": (mock_policy_value + ipw3.eps - ground_truth_policy_value) ** 2,
        }
    # check performance estimators
    ope_ = ContinuousOffPolicyEvaluation(
        bandit_feedback=synthetic_continuous_bandit_feedback, ope_estimators=[ipw, ipw3]
    )
    performance = ope_.evaluate_performance_of_estimators(
        ground_truth_policy_value=ground_truth_policy_value,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        action_by_evaluation_policy=action_by_evaluation_policy,
        metric=metric,
    )
    for k, v in performance.items():
        assert k in eval_metric_ope_dict, "Invalid key of performance response"
        assert v == eval_metric_ope_dict[k], "Invalid value of performance response"
    performance_df = ope_.summarize_estimators_comparison(
        ground_truth_policy_value=ground_truth_policy_value,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        action_by_evaluation_policy=action_by_evaluation_policy,
        metric=metric,
    )
    assert_frame_equal(
        performance_df, pd.DataFrame(eval_metric_ope_dict, index=[metric]).T
    ), "Invalid summarization (performance)"
