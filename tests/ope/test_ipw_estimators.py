import re

from conftest import generate_action_dist
import numpy as np
import pytest

from obp.ope import InverseProbabilityWeighting
from obp.ope import InverseProbabilityWeightingTuning
from obp.ope import SelfNormalizedInverseProbabilityWeighting
from obp.types import BanditFeedback


# lambda_, err, description
invalid_input_of_ipw_init = [
    (
        "",
        TypeError,
        r"`lambda_` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        None,
        TypeError,
        r"`lambda_` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (-1.0, ValueError, "`lambda_`= -1.0, must be >= 0.0."),
    (np.nan, ValueError, "lambda_ must not be nan"),
]


@pytest.mark.parametrize(
    "lambda_, err, description",
    invalid_input_of_ipw_init,
)
def test_ipw_init_using_invalid_inputs(
    lambda_,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = InverseProbabilityWeighting(lambda_=lambda_)


# lambdas, use_bias_upper_bound, delta, err, description
invalid_input_of_ipw_tuning_init = [
    (
        "",  #
        True,
        0.05,
        TypeError,
        "lambdas must be a list",
    ),
    (
        None,  #
        True,
        0.05,
        TypeError,
        "lambdas must be a list",
    ),
    (
        [""],  #
        True,
        0.05,
        TypeError,
        r"`an element of lambdas` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        [None],  #
        True,
        0.05,
        TypeError,
        r"`an element of lambdas` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (
        [],  #
        True,
        0.05,
        ValueError,
        "lambdas must not be empty",
    ),
    (
        [-1.0],  #
        True,
        0.05,
        ValueError,
        "`an element of lambdas`= -1.0, must be >= 0.0.",
    ),
    ([np.nan], True, 0.05, ValueError, "an element of lambdas must not be nan"),
    (
        [1],
        "",  #
        0.05,
        TypeError,
        "`use_bias_upper_bound` must be a bool",
    ),
    (
        [1],
        None,  #
        0.05,
        TypeError,
        "`use_bias_upper_bound` must be a bool",
    ),
    (
        [1],
        True,
        "",  #
        TypeError,
        "`delta` must be an instance of <class 'float'>",
    ),
    (
        [1],
        True,
        None,  #
        TypeError,
        "`delta` must be an instance of <class 'float'>",
    ),
    (
        [1],
        True,
        -1.0,  #
        ValueError,
        "`delta`= -1.0, must be >= 0.0.",
    ),
    (
        [1],
        True,
        1.1,  #
        ValueError,
        "`delta`= 1.1, must be <= 1.0.",
    ),
]


@pytest.mark.parametrize(
    "lambdas, use_bias_upper_bound, delta, err, description",
    invalid_input_of_ipw_tuning_init,
)
def test_ipw_tuning_init_using_invalid_inputs(
    lambdas,
    use_bias_upper_bound,
    delta,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = InverseProbabilityWeightingTuning(
            use_bias_upper_bound=use_bias_upper_bound, delta=delta, lambdas=lambdas
        )


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
        "action must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        None,  #
        np.ones(5),
        np.random.choice(3, size=5),
        "reward must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        None,  #
        np.random.choice(3, size=5),
        "pscore must be 1D array",
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
        "action must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros((3, 2), dtype=int),  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "action must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) + 8,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        r"action elements must be smaller than`",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        "4",  #
        np.ones(5),
        np.random.choice(3, size=5),
        "reward must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros((3, 2), dtype=int),  #
        np.ones(5),
        np.random.choice(3, size=5),
        "reward must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(4, dtype=int),  #
        np.ones(5),
        np.random.choice(3, size=5),
        "Expected `action.shape[0]",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        "4",  #
        np.random.choice(3, size=5),
        "pscore must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones((5, 3)),  #
        np.random.choice(3, size=5),
        "pscore must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(4),  #
        np.random.choice(3, size=5),
        "Expected `action.shape[0]",
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
