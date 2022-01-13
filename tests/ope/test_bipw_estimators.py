import re

from conftest import generate_action_dist
import numpy as np
import pytest

from obp.ope import BalancedInverseProbabilityWeighting
from obp.types import BanditFeedback


# lambda_, err, description
invalid_input_of_bipw_init = [
    (
        "",
        TypeError,
        r"lambda_ must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        None,
        TypeError,
        r"lambda_ must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (-1.0, ValueError, "lambda_ == -1.0, must be >= 0.0."),
    (np.nan, ValueError, "`lambda_` must not be nan"),
]


@pytest.mark.parametrize(
    "lambda_, err, description",
    invalid_input_of_bipw_init,
)
def test_bipw_init_using_invalid_inputs(
    lambda_,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = BalancedInverseProbabilityWeighting(
            lambda_=lambda_,
        )


# prepare bipw instances
bipw = BalancedInverseProbabilityWeighting()

# action_dist, action, reward, position, estimated_importance_weights, description
invalid_input_of_bipw = [
    (
        generate_action_dist(5, 4, 3),
        None,  #
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        np.ones(5),
        "`action` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        None,  #
        np.random.choice(3, size=5),
        np.ones(5),
        "`reward` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        None,  #
        "`estimated_importance_weights` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=float),  #
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        np.ones(5),
        "`action` elements must be integers in the range of",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) - 1,  #
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        np.ones(5),
        "`action` elements must be integers in the range of",
    ),
    (
        generate_action_dist(5, 4, 3),
        "4",  #
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        np.ones(5),
        "`action` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros((3, 2), dtype=int),  #
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        np.ones(5),
        "`action` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) + 8,  #
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        np.ones(5),
        r"`action` elements must be integers in the range of`",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        "4",  #
        np.random.choice(3, size=5),
        np.ones(5),
        "`reward` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros((3, 2), dtype=int),  #
        np.random.choice(3, size=5),
        np.ones(5),
        "`reward` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(4, dtype=int),  #
        np.random.choice(3, size=5),
        np.ones(5),
        "Expected `action.shape[0]",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        "4",  #
        "`estimated_importance_weights` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        np.ones((5, 3)),  #
        "`estimated_importance_weights` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        np.ones(4),  #
        "Expected `action.shape[0]",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        np.arange(5) - 1,  #
        "estimated_importance_weights must be non-negative",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.random.choice(3, size=5),
        None,  #
        "`estimated_importance_weights` must be 1D array",
    ),
]


@pytest.mark.parametrize(
    "action_dist, action, reward, position, estimated_importance_weights, description",
    invalid_input_of_bipw,
)
def test_bipw_using_invalid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    position: np.ndarray,
    estimated_importance_weights: np.ndarray,
    description: str,
) -> None:
    # prepare bipw instances
    bipw = BalancedInverseProbabilityWeighting()
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = bipw.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            position=position,
            estimated_importance_weights=estimated_importance_weights,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = bipw.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            position=position,
            estimated_importance_weights=estimated_importance_weights,
        )


def test_bipw_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the format of bipw variants using synthetic bandit data and random evaluation policy
    """
    action_dist = random_action_dist
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_dict["action_dist"] = action_dist
    # insert dummy values
    input_dict["estimated_importance_weights"] = np.ones(action_dist.shape[0])
    # check responce
    for estimator in [bipw]:
        estimated_policy_value = estimator.estimate_policy_value(**input_dict)
        assert isinstance(
            estimated_policy_value, float
        ), f"invalid type response: {estimator}"

    # make estimated_importance_weights too small (to check the boundedness of snbipw)
    input_dict["estimated_importance_weights"] = input_dict["pscore"] ** 3
    estimated_policy_value = bipw.estimate_policy_value(**input_dict)
    assert (
        estimated_policy_value <= 1
    ), f"estimated policy value of bipw should be smaller than or equal to 1 (because of its 1-boundedness), but the value is: {estimated_policy_value}"

    # remove necessary keys
    del input_dict["reward"]
    del input_dict["action"]
    del input_dict["estimated_importance_weights"]
    for estimator in [bipw]:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "estimate_policy_value() missing 3 required positional arguments: 'reward', 'action', and 'estimated_importance_weights'"
            ),
        ):
            _ = estimator.estimate_policy_value(**input_dict)
