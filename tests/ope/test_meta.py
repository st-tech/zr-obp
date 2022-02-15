from copy import deepcopy
from dataclasses import dataclass
import itertools
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from obp.ope import BaseOffPolicyEstimator
from obp.ope import DirectMethod
from obp.ope import OffPolicyEvaluation
from obp.types import BanditFeedback
from obp.utils import check_confidence_interval_arguments


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
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        position: array-like, shape (n_rounds,)
            Indices to differentiate positions in a recommendation interface where the actions are presented.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

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
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        position: array-like, shape (n_rounds,)
            Indices to differentiate positions in a recommendation interface where the actions are presented.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

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
        return mock_confidence_interval


@dataclass
class InverseProbabilityWeightingMock(BaseOffPolicyEstimator):
    """Inverse Probability Weighting (IPW) Mock."""

    estimator_name: str = "ipw"
    eps: float = 0.1

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

        position: array-like, shape (n_rounds,)
            Indices to differentiate positions in a recommendation interface where the actions are presented.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.

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

        position: array-like, shape (n_rounds,)
            Indices to differentiate positions in a recommendation interface where the actions are presented.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities
            by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

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


def test_meta_post_init(synthetic_bandit_feedback: BanditFeedback) -> None:
    """
    Test the __post_init__ function
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
    # __post__init__ raises RuntimeError when necessary_keys are not included in the bandit_feedback
    necessary_keys = ["action", "position", "reward"]
    for i in range(len(necessary_keys)):
        for deleted_keys in itertools.combinations(necessary_keys, i + 1):
            invalid_bandit_feedback_dict = {key: "_" for key in necessary_keys}
            # delete
            for k in deleted_keys:
                del invalid_bandit_feedback_dict[k]
            with pytest.raises(RuntimeError, match=r"Missing key*"):
                _ = OffPolicyEvaluation(
                    bandit_feedback=invalid_bandit_feedback_dict, ope_estimators=[ipw]
                )


def test_meta_estimated_rewards_by_reg_model_inputs(
    synthetic_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the estimate_policy_values/estimate_intervals functions wrt estimated_rewards_by_reg_model
    """
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[DirectMethod()]
    )

    action_dist = np.zeros(
        (synthetic_bandit_feedback["n_rounds"], synthetic_bandit_feedback["n_actions"])
    )
    with pytest.raises(ValueError):
        ope_.estimate_policy_values(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=None,
        )

    with pytest.raises(ValueError):
        ope_.estimate_intervals(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=None,
        )


# action_dist, estimated_rewards_by_reg_model, description
invalid_input_of_create_estimator_inputs = [
    (
        np.zeros((2, 3, 4)),
        np.zeros((2, 3, 3)),
        "Expected `estimated_rewards_by_reg_model.shape",
    ),
    (
        np.zeros((2, 3, 4)),
        {"dm": np.zeros((2, 3, 3))},
        r"Expected `estimated_rewards_by_reg_model\[dm\].shape",
    ),
    (
        np.zeros((2, 3, 4)),
        {"dm": None},
        r"`estimated_rewards_by_reg_model\[dm\]` must be 3D array",
    ),
    (np.zeros((2, 3)), None, "`action_dist` must be 3D array"),
    ("3", None, "`action_dist` must be 3D array"),
    (None, None, "`action_dist` must be 3D array"),
]

valid_input_of_create_estimator_inputs = [
    (
        np.zeros((2, 3, 4)),
        np.zeros((2, 3, 4)),
        "same shape",
    ),
    (
        np.zeros((2, 3, 4)),
        {"dm": np.zeros((2, 3, 4))},
        "same shape",
    ),
    (np.zeros((2, 3, 1)), None, "`estimated_rewards_by_reg_model` is None"),
]


@pytest.mark.parametrize(
    "action_dist, estimated_rewards_by_reg_model, description",
    invalid_input_of_create_estimator_inputs,
)
def test_meta_create_estimator_inputs_using_invalid_input_data(
    action_dist,
    estimated_rewards_by_reg_model,
    description: str,
    synthetic_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the _create_estimator_inputs using valid data
    """
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[ipw]
    )
    # raise ValueError when the shape of two arrays are different
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    # _create_estimator_inputs function is called in the following functions
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.estimate_policy_values(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.estimate_intervals(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.summarize_off_policy_estimates(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.evaluate_performance_of_estimators(
            ground_truth_policy_value=0.1,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ope_.summarize_estimators_comparison(
            ground_truth_policy_value=0.1,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )


@pytest.mark.parametrize(
    "action_dist, estimated_rewards_by_reg_model, description",
    valid_input_of_create_estimator_inputs,
)
def test_meta_create_estimator_inputs_using_valid_input_data(
    action_dist,
    estimated_rewards_by_reg_model,
    description: str,
    synthetic_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the _create_estimator_inputs using invalid data
    """
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[ipw]
    )
    estimator_inputs = ope_._create_estimator_inputs(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    assert set(estimator_inputs.keys()) == set(["ipw"])
    assert set(estimator_inputs["ipw"].keys()) == set(
        [
            "reward",
            "action",
            "pscore",
            "position",
            "action_dist",
            "estimated_rewards_by_reg_model",
            "estimated_pscore",
            "estimated_importance_weights",
            "p_e_a",
            "pi_b",
            "context",
            "action_embed",
        ]
    ), f"Invalid response of _create_estimator_inputs (test case: {description})"
    # _create_estimator_inputs function is called in the following functions
    _ = ope_.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    _ = ope_.estimate_intervals(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    _ = ope_.summarize_off_policy_estimates(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    _ = ope_.evaluate_performance_of_estimators(
        ground_truth_policy_value=0.1,
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    _ = ope_.summarize_estimators_comparison(
        ground_truth_policy_value=0.1,
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )


@pytest.mark.parametrize(
    "action_dist, estimated_rewards_by_reg_model, description",
    valid_input_of_create_estimator_inputs[:-1],
)
def test_meta_estimate_policy_values_using_valid_input_data(
    action_dist,
    estimated_rewards_by_reg_model,
    description: str,
    synthetic_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of estimate_policy_values using valid data
    """
    # single ope estimator
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[dm]
    )
    ope_.is_model_dependent = True
    assert ope_.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    ) == {
        "dm": mock_policy_value
    }, "OffPolicyEvaluation.estimate_policy_values ([DirectMethod]) returns a wrong value"
    # multiple ope estimators
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[dm, ipw]
    )
    ope_.is_model_dependent = True
    assert ope_.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    ) == {
        "dm": mock_policy_value,
        "ipw": mock_policy_value + ipw.eps,
    }, "OffPolicyEvaluation.estimate_policy_values ([DirectMethod, IPW]) returns a wrong value"


# alpha, n_bootstrap_samples, random_state, err, description
invalid_input_of_estimate_intervals = [
    (
        0.05,
        100,
        "s",
        ValueError,
        "'s' cannot be used to seed a numpy.random.RandomState instance",
    ),
    (0.05, -1, 1, ValueError, "n_bootstrap_samples == -1, must be >= 1"),
    (
        0.05,
        "s",
        1,
        TypeError,
        "n_bootstrap_samples must be an instance of <class 'int'>, not <class 'str'>",
    ),
    (-1.0, 1, 1, ValueError, "alpha == -1.0, must be >= 0.0"),
    (2.0, 1, 1, ValueError, "alpha == 2.0, must be <= 1.0"),
    (
        "0",
        1,
        1,
        TypeError,
        "alpha must be an instance of <class 'float'>, not <class 'str'>",
    ),
]

valid_input_of_estimate_intervals = [
    (0.05, 100, 1, "random_state is 1"),
    (0.05, 1, 1, "n_bootstrap_samples is 1"),
]


@pytest.mark.parametrize(
    "action_dist, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, err, description_2",
    invalid_input_of_estimate_intervals,
)
def test_meta_estimate_intervals_using_invalid_input_data(
    action_dist,
    estimated_rewards_by_reg_model,
    description_1: str,
    alpha,
    n_bootstrap_samples,
    random_state,
    err,
    description_2: str,
    synthetic_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of estimate_intervals using invalid data
    """
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[dm]
    )
    with pytest.raises(err, match=f"{description_2}*"):
        _ = ope_.estimate_intervals(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
    # estimate_intervals function is called in summarize_off_policy_estimates
    with pytest.raises(err, match=f"{description_2}*"):
        _ = ope_.summarize_off_policy_estimates(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "action_dist, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    valid_input_of_estimate_intervals,
)
def test_meta_estimate_intervals_using_valid_input_data(
    action_dist,
    estimated_rewards_by_reg_model,
    description_1: str,
    alpha: float,
    n_bootstrap_samples: int,
    random_state: int,
    description_2: str,
    synthetic_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of estimate_intervals using valid data
    """
    # single ope estimator
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[dm]
    )
    assert ope_.estimate_intervals(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    ) == {
        "dm": mock_confidence_interval
    }, "OffPolicyEvaluation.estimate_intervals ([DirectMethod]) returns a wrong value"
    # multiple ope estimators
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[dm, ipw]
    )
    assert ope_.estimate_intervals(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    ) == {
        "dm": mock_confidence_interval,
        "ipw": {k: v + ipw.eps for k, v in mock_confidence_interval.items()},
    }, "OffPolicyEvaluation.estimate_intervals ([DirectMethod, IPW]) returns a wrong value"


@pytest.mark.parametrize(
    "action_dist, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    valid_input_of_estimate_intervals,
)
def test_meta_summarize_off_policy_estimates(
    action_dist,
    estimated_rewards_by_reg_model,
    description_1: str,
    alpha: float,
    n_bootstrap_samples: int,
    random_state: int,
    description_2: str,
    synthetic_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of summarize_off_policy_estimates using valid data
    """
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[ipw, ipw3]
    )
    value, interval = ope_.summarize_off_policy_estimates(
        action_dist=action_dist,
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
        / synthetic_bandit_feedback["reward"].mean()
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
    zero_reward_bandit_feedback = deepcopy(synthetic_bandit_feedback)
    zero_reward_bandit_feedback["reward"] = np.zeros(
        zero_reward_bandit_feedback["reward"].shape[0]
    )
    ope_ = OffPolicyEvaluation(
        bandit_feedback=zero_reward_bandit_feedback, ope_estimators=[ipw, ipw3]
    )
    value, _ = ope_.summarize_off_policy_estimates(
        action_dist=action_dist,
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
    ("foo", 0.3, ValueError, "`metric` must be either 'relative-ee' or 'se'"),
    (
        "se",
        1,
        TypeError,
        "ground_truth_policy_value must be an instance of <class 'float'>, not <class 'int'>.",
    ),
    (
        "se",
        "a",
        TypeError,
        "ground_truth_policy_value must be an instance of <class 'float'>, not <class 'str'>.",
    ),
    (
        "relative-ee",
        0.0,
        ValueError,
        "`ground_truth_policy_value` must be non-zero when metric is relative-ee",
    ),
]

valid_input_of_evaluation_performance_of_estimators = [
    ("se", 0.0, "metric is se and ground_truth_policy_value is 0.0"),
    ("relative-ee", 1.0, "metric is relative-ee and ground_truth_policy_value is 1.0"),
]


@pytest.mark.parametrize(
    "action_dist, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "metric, ground_truth_policy_value, err, description_2",
    invalid_input_of_evaluation_performance_of_estimators,
)
def test_meta_evaluate_performance_of_estimators_using_invalid_input_data(
    action_dist,
    estimated_rewards_by_reg_model,
    description_1: str,
    metric,
    ground_truth_policy_value,
    err,
    description_2: str,
    synthetic_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of evaluate_performance_of_estimators using invalid data
    """
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[dm]
    )
    with pytest.raises(err, match=f"{description_2}*"):
        _ = ope_.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
            metric=metric,
        )
    # estimate_intervals function is called in summarize_off_policy_estimates
    with pytest.raises(err, match=f"{description_2}*"):
        _ = ope_.summarize_estimators_comparison(
            ground_truth_policy_value=ground_truth_policy_value,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
            metric=metric,
        )


@pytest.mark.parametrize(
    "action_dist, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "metric, ground_truth_policy_value, description_2",
    valid_input_of_evaluation_performance_of_estimators,
)
def test_meta_evaluate_performance_of_estimators_using_valid_input_data(
    action_dist,
    estimated_rewards_by_reg_model,
    description_1: str,
    metric,
    ground_truth_policy_value,
    description_2: str,
    synthetic_bandit_feedback: BanditFeedback,
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
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[ipw, ipw3]
    )
    performance = ope_.evaluate_performance_of_estimators(
        ground_truth_policy_value=ground_truth_policy_value,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        action_dist=action_dist,
        metric=metric,
    )
    for k, v in performance.items():
        assert k in eval_metric_ope_dict, "Invalid key of performance response"
        assert v == eval_metric_ope_dict[k], "Invalid value of performance response"
    performance_df = ope_.summarize_estimators_comparison(
        ground_truth_policy_value=ground_truth_policy_value,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        action_dist=action_dist,
        metric=metric,
    )
    assert_frame_equal(
        performance_df, pd.DataFrame(eval_metric_ope_dict, index=[metric]).T
    ), "Invalid summarization (performance)"
