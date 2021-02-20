import re

import pytest
import numpy as np

from obp.types import BanditFeedback
from obp.ope import (
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
)
from conftest import generate_action_dist

# prepare ipw instances
ipw = InverseProbabilityWeighting()
snipw = SelfNormalizedInverseProbabilityWeighting()


# action_dist, action, reward, pscore, position, description
invalid_input_of_ipw = [
    (
        generate_action_dist(5, 4, 3),
        None,
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        None,
        np.ones(5),
        np.random.choice(3, size=5),
        "reward must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        None,
        np.random.choice(3, size=5),
        "pscore must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=float),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) - 1,
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        "4",
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros((3, 2), dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action must be 1-dimensional",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) + 8,
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action elements must be smaller than the second dimension of action_dist",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        "4",
        np.ones(5),
        np.random.choice(3, size=5),
        "reward must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros((3, 2), dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "reward must be 1-dimensional",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(4, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action and reward must be the same size.",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        "4",
        np.random.choice(3, size=5),
        "pscore must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones((5, 3)),
        np.random.choice(3, size=5),
        "pscore must be 1-dimensional",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(4),
        np.random.choice(3, size=5),
        "action, reward, and pscore must be the same size.",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.arange(5),
        np.random.choice(3, size=5),
        "pscore must be positive",
    ),
]


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, description",
    invalid_input_of_ipw,
)
def test_ipw_using_invalid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    description: str,
) -> None:
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = snipw.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = snipw.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )


def test_ipw_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the format of ipw variants using synthetic bandit data and random evaluation policy
    """
    action_dist = random_action_dist
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_dict["action_dist"] = action_dist
    # ipw estimtors can be used without estimated_rewards_by_reg_model
    for estimator in [ipw, snipw]:
        estimated_policy_value = estimator.estimate_policy_value(**input_dict)
        assert isinstance(
            estimated_policy_value, float
        ), f"invalid type response: {estimator}"
    # remove necessary keys
    del input_dict["reward"]
    del input_dict["pscore"]
    del input_dict["action"]
    for estimator in [ipw, snipw]:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "estimate_policy_value() missing 3 required positional arguments: 'reward', 'action', and 'pscore'"
            ),
        ):
            _ = estimator.estimate_policy_value(**input_dict)


def test_boundedness_of_snipw_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the boundedness of snipw estimators using synthetic bandit data and random evaluation policy
    """
    action_dist = random_action_dist
    # prepare snipw
    snipw = SelfNormalizedInverseProbabilityWeighting()
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_dict["action_dist"] = action_dist
    # make pscore too small (to check the boundedness of snipw)
    input_dict["pscore"] = input_dict["pscore"] ** 3
    estimated_policy_value = snipw.estimate_policy_value(**input_dict)
    assert (
        estimated_policy_value <= 1
    ), f"estimated policy value of snipw should be smaller than or equal to 1 (because of its 1-boundedness), but the value is: {estimated_policy_value}"
