import re

import pytest
import numpy as np
import torch

from obp.types import BanditFeedback
from obp.ope import (
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    InverseProbabilityWeightingTuning,
)
from conftest import generate_action_dist


def test_ipw_init():
    # lambda_
    with pytest.raises(
        TypeError,
        match=r"`lambda_` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ):
        InverseProbabilityWeighting(lambda_=None)

    with pytest.raises(
        TypeError,
        match=r"`lambda_` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ):
        InverseProbabilityWeighting(lambda_="")

    with pytest.raises(ValueError, match=r"`lambda_`= -1.0, must be >= 0.0."):
        InverseProbabilityWeighting(lambda_=-1.0)

    with pytest.raises(ValueError, match=r"lambda_ must not be nan"):
        InverseProbabilityWeighting(lambda_=np.nan)

    # lambdas
    with pytest.raises(
        TypeError,
        match=r"`an element of lambdas` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ):
        InverseProbabilityWeightingTuning(lambdas=[None])

    with pytest.raises(
        TypeError,
        match=r"`an element of lambdas` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ):
        InverseProbabilityWeightingTuning(lambdas=[""])

    with pytest.raises(
        ValueError, match="`an element of lambdas`= -1.0, must be >= 0.0."
    ):
        InverseProbabilityWeightingTuning(lambdas=[-1.0])

    with pytest.raises(ValueError, match="an element of lambdas must not be nan"):
        InverseProbabilityWeightingTuning(lambdas=[np.nan])

    with pytest.raises(ValueError, match="lambdas must not be empty"):
        InverseProbabilityWeightingTuning(lambdas=[])

    with pytest.raises(TypeError, match="lambdas must be a list"):
        InverseProbabilityWeightingTuning(lambdas="")

    with pytest.raises(TypeError, match="lambdas must be a list"):
        InverseProbabilityWeightingTuning(lambdas=None)


# prepare ipw instances
ipw = InverseProbabilityWeighting()
snipw = SelfNormalizedInverseProbabilityWeighting()
ipw_tuning = InverseProbabilityWeightingTuning(lambdas=[10, 1000])


# action_dist, action, reward, pscore, position, description
invalid_input_of_ipw = [
    (
        generate_action_dist(5, 4, 3),
        None,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        None,  #
        np.ones(5),
        np.random.choice(3, size=5),
        "reward must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        None,  #
        np.random.choice(3, size=5),
        "pscore must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=float),  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) - 1,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        "4",  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros((3, 2), dtype=int),  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action must be 1-dimensional",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) + 8,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action elements must be smaller than the second dimension of action_dist",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        "4",  #
        np.ones(5),
        np.random.choice(3, size=5),
        "reward must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros((3, 2), dtype=int),  #
        np.ones(5),
        np.random.choice(3, size=5),
        "reward must be 1-dimensional",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(4, dtype=int),  #
        np.ones(5),
        np.random.choice(3, size=5),
        "action and reward must be the same size.",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        "4",  #
        np.random.choice(3, size=5),
        "pscore must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones((5, 3)),  #
        np.random.choice(3, size=5),
        "pscore must be 1-dimensional",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(4),  #
        np.random.choice(3, size=5),
        "action, reward, and pscore must be the same size.",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.arange(5),  #
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
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw_tuning.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw_tuning.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )


# action_dist, action, reward, pscore, position, description
invalid_input_tensor_of_ipw = [
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        None,  #
        torch.zeros(5, dtype=torch.float32),
        torch.ones(5),
        torch.from_numpy(np.random.choice(3, size=5)),
        "action must be Tensor",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.int64),
        None,  #
        torch.ones(5),
        torch.from_numpy(np.random.choice(3, size=5)),
        "reward must be Tensor",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.int64),
        torch.zeros(5, dtype=torch.float32),
        None,  #
        torch.from_numpy(np.random.choice(3, size=5)),
        "pscore must be Tensor",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.float64),  #
        torch.zeros(5, dtype=torch.float32),
        torch.ones(5),
        torch.from_numpy(np.random.choice(3, size=5)),
        "action elements must be non-negative integers",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.float64) - 1,  #
        torch.zeros(5, dtype=torch.float32),
        torch.ones(5),
        torch.from_numpy(np.random.choice(3, size=5)),
        "action elements must be non-negative integers",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        "4",  #
        torch.zeros(5, dtype=torch.float32),
        torch.ones(5),
        torch.from_numpy(np.random.choice(3, size=5)),
        "action must be Tensor",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros((3, 2), dtype=torch.int64),  #
        torch.zeros(5, dtype=torch.float32),
        torch.ones(5),
        torch.from_numpy(np.random.choice(3, size=5)),
        "action must be 1-dimensional",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.int64) + 8,  #
        torch.zeros(5, dtype=torch.float32),
        torch.ones(5),
        torch.from_numpy(np.random.choice(3, size=5)),
        "action elements must be smaller than the second dimension of action_dist",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.int64),
        "4",  #
        torch.ones(5),
        torch.from_numpy(np.random.choice(3, size=5)),
        "reward must be Tensor",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.int64),
        torch.zeros((3, 2), dtype=torch.float32),  #
        torch.ones(5),
        torch.from_numpy(np.random.choice(3, size=5)),
        "reward must be 1-dimensional",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.int64),
        torch.zeros(4, dtype=torch.float32),  #
        torch.ones(5),
        torch.from_numpy(np.random.choice(3, size=5)),
        "action and reward must be the same size.",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.int64),
        torch.zeros(5, dtype=torch.float32),
        "4",  #
        torch.from_numpy(np.random.choice(3, size=5)),
        "pscore must be Tensor",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.int64),
        torch.zeros(5, dtype=torch.float32),
        torch.ones((5, 3)),  #
        torch.from_numpy(np.random.choice(3, size=5)),
        "pscore must be 1-dimensional",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.int64),
        torch.zeros(5, dtype=torch.float32),
        torch.ones(4),  #
        torch.from_numpy(np.random.choice(3, size=5)),
        "action, reward, and pscore must be the same size.",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.zeros(5, dtype=torch.int64),
        torch.zeros(5, dtype=torch.float32),
        torch.from_numpy(np.arange(5)),  #
        torch.from_numpy(np.random.choice(3, size=5)),
        "pscore must be positive",
    ),
]


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, description",
    invalid_input_tensor_of_ipw,
)
def test_ipw_using_invalid_input_tensor_data(
    action_dist: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    pscore: torch.Tensor,
    position: torch.Tensor,
    description: str,
) -> None:
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw.estimate_policy_value_tensor(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = snipw.estimate_policy_value_tensor(
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
    # ipw estimators can be used without estimated_rewards_by_reg_model
    for estimator in [ipw, snipw, ipw_tuning]:
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

    input_tensor_dict = {
        k: v if v is None else torch.from_numpy(v)
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_tensor_dict["action_dist"] = torch.from_numpy(action_dist)
    for estimator in [ipw, snipw]:
        estimated_policy_value = estimator.estimate_policy_value_tensor(
            **input_tensor_dict
        )
        assert isinstance(
            estimated_policy_value, torch.Tensor
        ), f"invalid type response: {estimator}"
    # remove necessary keys
    del input_tensor_dict["reward"]
    del input_tensor_dict["pscore"]
    del input_tensor_dict["action"]
    for estimator in [ipw, snipw]:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "estimate_policy_value_tensor() missing 3 required positional arguments: 'reward', 'action', and 'pscore'"
            ),
        ):
            _ = estimator.estimate_policy_value_tensor(**input_tensor_dict)


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

    input_tensor_dict = {
        k: v if v is None else torch.from_numpy(v)
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_tensor_dict["action_dist"] = torch.from_numpy(action_dist)
    # make pscore too small (to check the boundedness of snipw)
    input_tensor_dict["pscore"] = input_tensor_dict["pscore"] ** 3
    estimated_policy_value_tensor = snipw.estimate_policy_value_tensor(
        **input_tensor_dict
    )
    assert (
        estimated_policy_value_tensor.item() <= 1
    ), f"estimated policy value of snipw should be smaller than or equal to 1 (because of its 1-boundedness), but the value is: {estimated_policy_value_tensor.item()}"
