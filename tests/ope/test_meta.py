from typing import Dict, Optional, Union
from dataclasses import dataclass
import itertools
from copy import deepcopy

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import torch

from obp.types import BanditFeedback
from obp.ope import OffPolicyEvaluation, BaseOffPolicyEstimator
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
        position: Union[np.ndarray, torch.Tensor],
        action_dist: Union[np.ndarray, torch.Tensor],
        estimated_rewards_by_reg_model: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> Union[float, torch.Tensor]:
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
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        mock_policy_value: float
        """
        return mock_policy_value

    def estimate_policy_value_tensor(
        self,
        position: torch.Tensor,
        action_dist: torch.Tensor,
        estimated_rewards_by_reg_model: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Estimate the policy value of evaluation policy and return PyTorch Tensor.
        This is intended for being used with NNPolicyLearner.

        Parameters
        ----------
        position: Tensor, shape (n_rounds,)
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

        action_dist: Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: Tensor, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        Returns
        ----------
        mock_policy_value: Tensor
        """
        return torch.Tensor(mock_policy_value)

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
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in the bootstrap procedure.

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
        **kwargs,
    ) -> np.ndarray:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        mock_policy_value: float

        """
        return mock_policy_value + self.eps

    def estimate_policy_value_tensor(
        self,
        reward: torch.Tensor,
        action: torch.Tensor,
        position: torch.Tensor,
        pscore: torch.Tensor,
        action_dist: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Estimate the policy value of evaluation policy and return PyTorch Tensor.
        This is intended for being used with NNPolicyLearner.

        Parameters
        ----------
        reward: Tensor, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        action: Tensor, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: Tensor, shape (n_rounds,)
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

        pscore: Tensor, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: Tensor, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        mock_policy_value: Tensor

        """
        return torch.Tensor(mock_policy_value + self.eps)

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
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,)
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities
            by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        alpha: float, default=0.05
            Significance level.

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
    necessary_keys = ["action", "position", "reward", "pscore"]
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


# action_dist, estimated_rewards_by_reg_model, description
invalid_input_of_create_estimator_inputs = [
    (
        np.zeros((2, 3, 4)),
        np.zeros((2, 3, 3)),
        "estimated_rewards_by_reg_model.shape must be the same as action_dist.shape",
    ),
    (
        np.zeros((2, 3, 4)),
        {"dm": np.zeros((2, 3, 3))},
        r"estimated_rewards_by_reg_model\[dm\].shape must be the same as action_dist.shape",
    ),
    (
        np.zeros((2, 3, 4)),
        {"dm": None},
        r"estimated_rewards_by_reg_model\[dm\] must be ndarray",
    ),
    (np.zeros((2, 3)), None, "action_dist.ndim must be 3-dimensional"),
    ("3", None, "action_dist must be ndarray"),
    (None, None, "action_dist must be ndarray"),
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
    (np.zeros((2, 3, 1)), None, "estimated_rewards_by_reg_model is None"),
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
    valid_input_of_create_estimator_inputs,
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
    assert ope_.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    ) == {
        "dm": mock_policy_value,
        "ipw": mock_policy_value + ipw.eps,
    }, "OffPolicyEvaluation.estimate_policy_values ([DirectMethod, IPW]) returns a wrong value"


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
    "action_dist, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    invalid_input_of_estimate_intervals,
)
def test_meta_estimate_intervals_using_invalid_input_data(
    action_dist,
    estimated_rewards_by_reg_model,
    description_1: str,
    alpha,
    n_bootstrap_samples,
    random_state,
    description_2: str,
    synthetic_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of estimate_intervals using invalid data
    """
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[dm]
    )
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = ope_.estimate_intervals(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
    # estimate_intervals function is called in summarize_off_policy_estimates
    with pytest.raises(ValueError, match=f"{description_2}*"):
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
    "action_dist, estimated_rewards_by_reg_model, description_1",
    valid_input_of_create_estimator_inputs,
)
@pytest.mark.parametrize(
    "metric, ground_truth_policy_value, description_2",
    invalid_input_of_evaluation_performance_of_estimators,
)
def test_meta_evaluate_performance_of_estimators_using_invalid_input_data(
    action_dist,
    estimated_rewards_by_reg_model,
    description_1: str,
    metric,
    ground_truth_policy_value,
    description_2: str,
    synthetic_bandit_feedback: BanditFeedback,
) -> None:
    """
    Test the response of evaluate_performance_of_estimators using invalid data
    """
    ope_ = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=[dm]
    )
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = ope_.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_dist=action_dist,
            metric=metric,
        )
    # estimate_intervals function is called in summarize_off_policy_estimates
    with pytest.raises(ValueError, match=f"{description_2}*"):
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
