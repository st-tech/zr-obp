from typing import Dict, Optional
from dataclasses import dataclass
import itertools
from copy import deepcopy
import re

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from obp.types import BanditFeedback
from obp.ope import (
    SlateOffPolicyEvaluation,
    SlateStandardIPS,
    SlateIndependentIPS,
    SlateRewardInteractionIPS,
)
from obp.utils import check_confidence_interval_arguments


mock_policy_value = 0.5
mock_confidence_interval = {
    "mean": 0.5,
    "95.0% CI (lower)": 0.3,
    "95.0% CI (upper)": 0.7,
}


@dataclass
class SlateStandardIPSMock(SlateStandardIPS):
    """Slate Standard Inverse Propensity Scoring (SIPS) Mock."""

    estimator_name: str = "sips"
    eps: float = 0.1

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Returns
        ----------
        mock_policy_value: float

        """
        return mock_policy_value + self.eps

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
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
class SlateIndependentIPSMock(SlateIndependentIPS):
    """Slate Independent Inverse Propensity Scoring (IIPS) Mock."""

    estimator_name: str = "iips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Returns
        ----------
        mock_policy_value: float

        """
        return mock_policy_value

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_item_position: np.ndarray,
        evaluation_policy_pscore_item_position: np.ndarray,
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


@dataclass
class SlateRewardInteractionIPSMock(SlateRewardInteractionIPS):
    """Slate Recursive Inverse Propensity Scoring (RIPS) Mock."""

    estimator_name: str = "rips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Returns
        ----------
        mock_policy_value: float

        """
        return mock_policy_value

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
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
sips = SlateStandardIPSMock(len_list=3)
sips2 = SlateStandardIPSMock(len_list=3, eps=0.02)
sips3 = SlateStandardIPSMock(len_list=3, estimator_name="sips3")
iips = SlateIndependentIPSMock(len_list=3)
rips = SlateRewardInteractionIPSMock(len_list=3)


def test_meta_post_init(synthetic_slate_bandit_feedback: BanditFeedback) -> None:
    """
    Test the __post_init__ function
    """
    # __post_init__ saves the latter estimator when the same estimator name is used
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[sips, sips2]
    )
    assert ope_.ope_estimators_ == {
        "sips": sips2
    }, "__post_init__ returns a wrong value"
    # __post_init__ can handle the same estimator if the estimator names are different
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[sips, sips3]
    )
    assert ope_.ope_estimators_ == {
        "sips": sips,
        "sips3": sips3,
    }, "__post_init__ returns a wrong value"
    # __post__init__ raises RuntimeError when necessary_keys are not included in the bandit_feedback
    necessary_keys = ["slate_id", "position", "reward"]
    for i in range(len(necessary_keys)):
        for deleted_keys in itertools.combinations(necessary_keys, i + 1):
            invalid_bandit_feedback_dict = {key: "_" for key in necessary_keys}
            # delete
            for k in deleted_keys:
                del invalid_bandit_feedback_dict[k]
            with pytest.raises(RuntimeError, match=r"Missing key*"):
                _ = SlateOffPolicyEvaluation(
                    bandit_feedback=invalid_bandit_feedback_dict, ope_estimators=[sips]
                )


# evaluation_policy_pscore, description
invalid_input_of_create_estimator_inputs = [
    (
        None,
        "one of evaluation_policy_pscore, evaluation_policy_pscore_item_position, or evaluation_policy_pscore_cascade must be given",
    ),
]

# evaluation_policy_pscore, evaluation_policy_pscore_item_position, evaluation_policy_pscore_cascade, description
valid_input_of_create_estimator_inputs = [
    (
        np.ones(300),
        np.ones(300),
        np.ones(300),
        "deterministic evaluation policy",
    ),
]


@pytest.mark.parametrize(
    "evaluation_policy_pscore, description",
    invalid_input_of_create_estimator_inputs,
)
def test_meta_create_estimator_inputs_using_invalid_input_data(
    evaluation_policy_pscore,
    description: str,
    synthetic_slate_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the _create_estimator_inputs using valid data and a sips estimator
    """
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[sips]
    )
    # raise ValueError when the shape of two arrays are different
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_._create_estimator_inputs(
            evaluation_policy_pscore=evaluation_policy_pscore
        )
    # _create_estimator_inputs function is called in the following functions
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.estimate_policy_values(
            evaluation_policy_pscore=evaluation_policy_pscore
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.estimate_intervals(evaluation_policy_pscore=evaluation_policy_pscore)
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.summarize_off_policy_estimates(
            evaluation_policy_pscore=evaluation_policy_pscore
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.evaluate_performance_of_estimators(
            ground_truth_policy_value=0.1,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.summarize_estimators_comparison(
            ground_truth_policy_value=0.1,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )


@pytest.mark.parametrize(
    "evaluation_policy_pscore, evaluation_policy_pscore_item_position, evaluation_policy_pscore_cascade, description",
    valid_input_of_create_estimator_inputs,
)
def test_meta_create_estimator_inputs_using_valid_input_data(
    evaluation_policy_pscore,
    evaluation_policy_pscore_item_position,
    evaluation_policy_pscore_cascade,
    description: str,
    synthetic_slate_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the _create_estimator_inputs using invalid data
    """
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[sips]
    )
    estimator_inputs = ope_._create_estimator_inputs(
        evaluation_policy_pscore=evaluation_policy_pscore
    )
    assert set(estimator_inputs.keys()) == set(
        [
            "reward",
            "pscore",
            "pscore_item_position",
            "pscore_cascade",
            "position",
            "evaluation_policy_pscore",
            "evaluation_policy_pscore_item_position",
            "evaluation_policy_pscore_cascade",
            "slate_id",
        ]
    ), f"Invalid response of _create_estimator_inputs (test case: {description})"
    # _create_estimator_inputs function is called in the following functions
    _ = ope_.estimate_policy_values(evaluation_policy_pscore=evaluation_policy_pscore)
    _ = ope_.estimate_intervals(evaluation_policy_pscore=evaluation_policy_pscore)
    _ = ope_.summarize_off_policy_estimates(
        evaluation_policy_pscore=evaluation_policy_pscore
    )
    _ = ope_.evaluate_performance_of_estimators(
        ground_truth_policy_value=0.1, evaluation_policy_pscore=evaluation_policy_pscore
    )
    _ = ope_.summarize_estimators_comparison(
        ground_truth_policy_value=0.1, evaluation_policy_pscore=evaluation_policy_pscore
    )


@pytest.mark.parametrize(
    "evaluation_policy_pscore, evaluation_policy_pscore_item_position, evaluation_policy_pscore_cascade, description",
    valid_input_of_create_estimator_inputs,
)
def test_meta_estimate_policy_values_using_valid_input_data(
    evaluation_policy_pscore,
    evaluation_policy_pscore_item_position,
    evaluation_policy_pscore_cascade,
    description: str,
    synthetic_slate_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of estimate_policy_values using valid data
    """
    # single ope estimator (iips)
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[iips]
    )
    assert ope_.estimate_policy_values(
        evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position
    ) == {
        "iips": mock_policy_value
    }, "SlateOffPolicyEvaluation.estimate_policy_values ([IIPS]) returns a wrong value"
    # multiple ope estimators
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback,
        ope_estimators=[iips, sips, rips],
    )
    assert ope_.estimate_policy_values(
        evaluation_policy_pscore=evaluation_policy_pscore,
        evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
    ) == {
        "iips": mock_policy_value,
        "sips": mock_policy_value + sips.eps,
        "rips": mock_policy_value,
    }, "SlateOffPolicyEvaluation.estimate_policy_values ([IIPS, SIPS, RIPS]) returns a wrong value"


@pytest.mark.parametrize(
    "evaluation_policy_pscore, evaluation_policy_pscore_item_position, evaluation_policy_pscore_cascade, description",
    valid_input_of_create_estimator_inputs,
)
def test_meta_estimate_policy_values_using_various_pscores(
    evaluation_policy_pscore,
    evaluation_policy_pscore_item_position,
    evaluation_policy_pscore_cascade,
    description: str,
    synthetic_slate_bandit_feedback: BanditFeedback,
) -> None:
    necessary_keys = [
        "reward",
        "position",
        "evaluation_policy_pscore",
        "evaluation_policy_pscore_item_position",
        "evaluation_policy_pscore_cascade" "slate_id",
    ]
    pscore_keys = [
        "pscore",
        "pscore_item_position",
        "pscore_cascade",
    ]
    # TypeError must be raised when required positional arguments are missing
    for i in range(len(necessary_keys)):
        for deleted_keys in itertools.combinations(pscore_keys, i + 1):
            copied_feedback = deepcopy(synthetic_slate_bandit_feedback)
            # delete
            for k in deleted_keys:
                del copied_feedback[k]
            with pytest.raises(
                TypeError,
                match=re.escape("estimate_policy_value() missing"),
            ):
                ope_ = SlateOffPolicyEvaluation(
                    bandit_feedback=copied_feedback,
                    ope_estimators=[sips, iips, rips],
                )
                _ = ope_.estimate_policy_values(
                    evaluation_policy_pscore=evaluation_policy_pscore,
                    evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
                    evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
                )
    # pscore_item_position and evaluation_policy_pscore_item_position are not necessary when iips is not evaluated
    copied_feedback = deepcopy(synthetic_slate_bandit_feedback)
    del copied_feedback["pscore_item_position"]
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=copied_feedback,
        ope_estimators=[sips, rips],
    )
    _ = ope_.estimate_policy_values(
        evaluation_policy_pscore=evaluation_policy_pscore,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
    )


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
    "evaluation_policy_pscore, evaluation_policy_pscore_item_position, evaluation_policy_pscore_cascade, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    invalid_input_of_estimate_intervals,
)
def test_meta_estimate_intervals_using_invalid_input_data(
    evaluation_policy_pscore,
    evaluation_policy_pscore_item_position,
    evaluation_policy_pscore_cascade,
    description_1: str,
    alpha,
    n_bootstrap_samples,
    random_state,
    description_2: str,
    synthetic_slate_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of estimate_intervals using invalid data
    """
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[iips]
    )
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = ope_.estimate_intervals(
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
    # estimate_intervals function is called in summarize_off_policy_estimates
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = ope_.summarize_off_policy_estimates(
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "evaluation_policy_pscore, evaluation_policy_pscore_item_position, evaluation_policy_pscore_cascade, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    valid_input_of_estimate_intervals,
)
def test_meta_estimate_intervals_using_valid_input_data(
    evaluation_policy_pscore,
    evaluation_policy_pscore_item_position,
    evaluation_policy_pscore_cascade,
    description_1: str,
    alpha: float,
    n_bootstrap_samples: int,
    random_state: int,
    description_2: str,
    synthetic_slate_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of estimate_intervals using valid data
    """
    # single ope estimator
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[iips]
    )
    assert ope_.estimate_intervals(
        evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    ) == {
        "iips": mock_confidence_interval
    }, "SlateOffPolicyEvaluation.estimate_intervals ([IIPS]) returns a wrong value"
    # multiple ope estimators
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[iips, sips]
    )
    assert ope_.estimate_intervals(
        evaluation_policy_pscore=evaluation_policy_pscore,
        evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    ) == {
        "iips": mock_confidence_interval,
        "sips": {k: v + sips.eps for k, v in mock_confidence_interval.items()},
    }, "SlateOffPolicyEvaluation.estimate_intervals ([IIPS, SIPS]) returns a wrong value"


@pytest.mark.parametrize(
    "evaluation_policy_pscore, evaluation_policy_pscore_item_position, evaluation_policy_pscore_cascade, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    valid_input_of_estimate_intervals,
)
def test_meta_summarize_off_policy_estimates(
    evaluation_policy_pscore,
    evaluation_policy_pscore_item_position,
    evaluation_policy_pscore_cascade,
    description_1: str,
    alpha: float,
    n_bootstrap_samples: int,
    random_state: int,
    description_2: str,
    synthetic_slate_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of summarize_off_policy_estimates using valid data
    """
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[sips, sips3]
    )
    value, interval = ope_.summarize_off_policy_estimates(
        evaluation_policy_pscore=evaluation_policy_pscore,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )
    expected_value = pd.DataFrame(
        {
            "sips": mock_policy_value + sips.eps,
            "sips3": mock_policy_value + sips3.eps,
        },
        index=["estimated_policy_value"],
    ).T
    expected_value["relative_estimated_policy_value"] = expected_value[
        "estimated_policy_value"
    ] / (
        synthetic_slate_bandit_feedback["reward"].sum()
        / np.unique(synthetic_slate_bandit_feedback["slate_id"]).shape[0]
    )
    expected_interval = pd.DataFrame(
        {
            "sips": {k: v + sips.eps for k, v in mock_confidence_interval.items()},
            "sips3": {k: v + sips3.eps for k, v in mock_confidence_interval.items()},
        }
    ).T
    assert_frame_equal(value, expected_value), "Invalid summarization (policy value)"
    assert_frame_equal(interval, expected_interval), "Invalid summarization (interval)"
    # check relative estimated policy value when the average of bandit_feedback["reward"] is zero
    zero_reward_bandit_feedback = deepcopy(synthetic_slate_bandit_feedback)
    zero_reward_bandit_feedback["reward"] = np.zeros(
        zero_reward_bandit_feedback["reward"].shape[0]
    )
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=zero_reward_bandit_feedback, ope_estimators=[sips, sips3]
    )
    value, _ = ope_.summarize_off_policy_estimates(
        evaluation_policy_pscore=evaluation_policy_pscore,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )
    expected_value = pd.DataFrame(
        {
            "sips": mock_policy_value + sips.eps,
            "sips3": mock_policy_value + sips3.eps,
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
    "evaluation_policy_pscore, evaluation_policy_pscore_item_position, evaluation_policy_pscore_cascade, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "metric, ground_truth_policy_value, description_2",
    invalid_input_of_evaluation_performance_of_estimators,
)
def test_meta_evaluate_performance_of_estimators_using_invalid_input_data(
    evaluation_policy_pscore,
    evaluation_policy_pscore_item_position,
    evaluation_policy_pscore_cascade,
    description_1: str,
    metric,
    ground_truth_policy_value,
    description_2: str,
    synthetic_slate_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of evaluate_performance_of_estimators using invalid data
    """
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[iips]
    )
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = ope_.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            metric=metric,
        )
    # estimate_intervals function is called in summarize_off_policy_estimates
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = ope_.summarize_estimators_comparison(
            ground_truth_policy_value=ground_truth_policy_value,
            evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
            metric=metric,
        )


@pytest.mark.parametrize(
    "evaluation_policy_pscore, evaluation_policy_pscore_item_position, evaluation_policy_pscore_cascade, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "metric, ground_truth_policy_value, description_2",
    valid_input_of_evaluation_performance_of_estimators,
)
def test_meta_evaluate_performance_of_estimators_using_valid_input_data(
    evaluation_policy_pscore,
    evaluation_policy_pscore_item_position,
    evaluation_policy_pscore_cascade,
    description_1: str,
    metric,
    ground_truth_policy_value,
    description_2: str,
    synthetic_slate_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of evaluate_performance_of_estimators using valid data
    """
    if metric == "relative-ee":
        # calculate relative-ee
        eval_metric_ope_dict = {
            "sips": np.abs(
                (mock_policy_value + sips.eps - ground_truth_policy_value)
                / ground_truth_policy_value
            ),
            "sips3": np.abs(
                (mock_policy_value + sips3.eps - ground_truth_policy_value)
                / ground_truth_policy_value
            ),
        }
    else:
        # calculate se
        eval_metric_ope_dict = {
            "sips": (mock_policy_value + sips.eps - ground_truth_policy_value) ** 2,
            "sips3": (mock_policy_value + sips3.eps - ground_truth_policy_value) ** 2,
        }
    # check performance estimators
    ope_ = SlateOffPolicyEvaluation(
        bandit_feedback=synthetic_slate_bandit_feedback, ope_estimators=[sips, sips3]
    )
    performance = ope_.evaluate_performance_of_estimators(
        ground_truth_policy_value=ground_truth_policy_value,
        evaluation_policy_pscore=evaluation_policy_pscore,
        metric=metric,
    )
    for k, v in performance.items():
        assert k in eval_metric_ope_dict, "Invalid key of performance response"
        assert v == eval_metric_ope_dict[k], "Invalid value of performance response"
    performance_df = ope_.summarize_estimators_comparison(
        ground_truth_policy_value=ground_truth_policy_value,
        evaluation_policy_pscore=evaluation_policy_pscore,
        metric=metric,
    )
    assert_frame_equal(
        performance_df, pd.DataFrame(eval_metric_ope_dict, index=[metric]).T
    ), "Invalid summarization (performance)"
