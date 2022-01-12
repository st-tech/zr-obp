import re

from conftest import generate_action_dist
import numpy as np
import pytest

from obp.ope import DirectMethod
from obp.types import BanditFeedback


# action_dist, position, estimated_rewards_by_reg_model, description
invalid_input_of_dm = [
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros((5, 4, 2)),  #
        "Expected `estimated_rewards_by_reg_model.shape == action_dist.shape`, but found it False",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        None,  #
        "`estimated_rewards_by_reg_model` must be 3D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        "4",  #
        "`estimated_rewards_by_reg_model` must be 3D array",
    ),
]


@pytest.mark.parametrize(
    "action_dist, position, estimated_rewards_by_reg_model, description",
    invalid_input_of_dm,
)
def test_dm_using_invalid_input_data(
    action_dist: np.ndarray,
    position: np.ndarray,
    estimated_rewards_by_reg_model: np.ndarray,
    description: str,
) -> None:
    dm = DirectMethod()
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = dm.estimate_policy_value(
            action_dist=action_dist,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = dm.estimate_interval(
            action_dist=action_dist,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )


def test_dm_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the performance of the direct method using synthetic bandit data and random evaluation policy
    """
    expected_reward = synthetic_bandit_feedback["expected_reward"][:, :, np.newaxis]
    action_dist = random_action_dist
    # compute ground truth policy value using expected reward
    q_pi_e = np.average(expected_reward[:, :, 0], weights=action_dist[:, :, 0], axis=1)
    # compute statistics of ground truth policy value
    gt_mean = q_pi_e.mean()
    # prepare dm
    dm = DirectMethod()
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_dict["action_dist"] = action_dist
    # estimated_rewards_by_reg_model is required
    with pytest.raises(
        TypeError,
        match=re.escape(
            "estimate_policy_value() missing 1 required positional argument: 'estimated_rewards_by_reg_model'"
        ),
    ):
        _ = dm.estimate_policy_value(**input_dict)
    # add estimated_rewards_by_reg_model
    input_dict["estimated_rewards_by_reg_model"] = expected_reward
    # check expectation
    estimated_policy_value = dm.estimate_policy_value(**input_dict)
    assert (
        gt_mean == estimated_policy_value
    ), "DM should be perfect when the regression model is perfect"
    # remove unnecessary keys
    del input_dict["reward"]
    del input_dict["pscore"]
    del input_dict["action"]
    estimated_policy_value = dm.estimate_policy_value(**input_dict)
    assert (
        gt_mean == estimated_policy_value
    ), "DM should be perfect when the regression model is perfect"
