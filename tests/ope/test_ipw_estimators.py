import re

import pytest
import numpy as np

from obp.types import BanditFeedback
from obp.ope import (
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
)


# prepare ipw instances
ipw = InverseProbabilityWeighting()
snipw = SelfNormalizedInverseProbabilityWeighting()


def test_ipw_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the format of ipw-like estimators using synthetic bandit data and random evaluation policy
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
    # remove used keys
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


def test_snipw_range_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the range of snipw estimators using synthetic bandit data and random evaluation policy
    """
    action_dist = random_action_dist
    # prepare dm
    ipw = InverseProbabilityWeighting()
    snipw = SelfNormalizedInverseProbabilityWeighting()
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_dict["action_dist"] = action_dist
    # make pscore too small (to check the normalization effect)
    input_dict["pscore"] = input_dict["pscore"] ** 3
    estimated_policy_value = ipw.estimate_policy_value(**input_dict)
    assert (
        estimated_policy_value > 1
    ), f"estimated policy value of ipw should be greater than 1 when pscore is too small, but {estimated_policy_value}"
    estimated_policy_value = snipw.estimate_policy_value(**input_dict)
    assert (
        estimated_policy_value <= 1
    ), f"estimated policy value of snipw should not be greater than 1 even if pscore is too small, but {estimated_policy_value}"
