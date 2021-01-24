import re

import pytest
import numpy as np

from obp.types import BanditFeedback
from obp.ope import DirectMethod


def test_dm_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the performance of dm-like estimators using synthetic bandit data and random evaluation policy
    """
    expected_reward = np.expand_dims(
        synthetic_bandit_feedback["expected_reward"], axis=-1
    )
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
    ), "DM should return gt mean when action_dist and reward function are both true"
    # remove unused keys
    del input_dict["reward"]
    del input_dict["pscore"]
    del input_dict["action"]
    estimated_policy_value = dm.estimate_policy_value(**input_dict)
    assert (
        gt_mean == estimated_policy_value
    ), "DM should return gt mean when action_dist and reward function are both true"
