import re

from conftest import generate_action_dist
import numpy as np
import pytest

from obp.ope import DirectMethod
from obp.ope import DoublyRobust
from obp.ope import DoublyRobustTuning
from obp.ope import DoublyRobustWithShrinkage
from obp.ope import DoublyRobustWithShrinkageTuning
from obp.ope import SelfNormalizedDoublyRobust
from obp.ope import SwitchDoublyRobust
from obp.ope import SwitchDoublyRobustTuning
from obp.types import BanditFeedback


# lambda_, err, description
invalid_input_of_dr_init = [
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
    invalid_input_of_dr_init,
)
def test_dr_init_using_invalid_inputs(
    lambda_,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = DoublyRobust(lambda_=lambda_)

    with pytest.raises(err, match=f"{description}*"):
        _ = SwitchDoublyRobust(lambda_=lambda_)

    with pytest.raises(err, match=f"{description}*"):
        _ = DoublyRobustWithShrinkage(lambda_=lambda_)


# lambdas, use_bias_upper_bound, delta, err, description
invalid_input_of_dr_tuning_init = [
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
    invalid_input_of_dr_tuning_init,
)
def test_dr_tuning_init_using_invalid_inputs(
    lambdas,
    use_bias_upper_bound,
    delta,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = DoublyRobustTuning(
            use_bias_upper_bound=use_bias_upper_bound, delta=delta, lambdas=lambdas
        )

    with pytest.raises(err, match=f"{description}*"):
        _ = SwitchDoublyRobustTuning(
            use_bias_upper_bound=use_bias_upper_bound,
            delta=delta,
            lambdas=lambdas,
        )

    with pytest.raises(err, match=f"{description}*"):
        _ = DoublyRobustWithShrinkageTuning(
            use_bias_upper_bound=use_bias_upper_bound,
            delta=delta,
            lambdas=lambdas,
        )


valid_input_of_dr_init = [
    (np.inf, "infinite lambda_"),
    (3.0, "float lambda_"),
    (2, "integer lambda_"),
]


@pytest.mark.parametrize(
    "lambda_, description",
    valid_input_of_dr_init,
)
def test_dr_init_using_valid_input_data(lambda_: float, description: str) -> None:
    _ = DoublyRobust(lambda_=lambda_)
    _ = DoublyRobustWithShrinkage(lambda_=lambda_)
    _ = SwitchDoublyRobust(lambda_=lambda_)


valid_input_of_dr_tuning_init = [
    ([3.0, np.inf, 100.0], "float lambda_"),
    ([2], "integer lambda_"),
]


@pytest.mark.parametrize(
    "lambdas, description",
    valid_input_of_dr_tuning_init,
)
def test_dr_tuning_init_using_valid_input_data(lambdas, description):
    _ = DoublyRobustTuning(lambdas=lambdas)
    _ = DoublyRobustWithShrinkageTuning(
        lambdas=lambdas,
    )
    _ = SwitchDoublyRobustTuning(
        lambdas=lambdas,
    )


# prepare instances
dm = DirectMethod()
dr = DoublyRobust()
dr_tuning = DoublyRobustTuning(lambdas=[1, 100], estimator_name="dr_tuning")
dr_os_0 = DoublyRobustWithShrinkage(lambda_=0.0)
dr_os_tuning = DoublyRobustWithShrinkageTuning(
    lambdas=[1, 100], estimator_name="dr_os_tuning"
)
dr_os_max = DoublyRobustWithShrinkage(lambda_=np.inf)
sndr = SelfNormalizedDoublyRobust()
switch_dr_0 = SwitchDoublyRobust(lambda_=0.0)
switch_dr_tuning = SwitchDoublyRobustTuning(
    lambdas=[1, 100], estimator_name="switch_dr_tuning"
)
switch_dr_max = SwitchDoublyRobust(lambda_=np.inf)

dr_estimators = [
    dr,
    dr_tuning,
    dr_os_0,
    dr_os_tuning,
    sndr,
    switch_dr_0,
    switch_dr_tuning,
]


# dr and self-normalized dr
# action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, description
invalid_input_of_dr = [
    (
        generate_action_dist(5, 4, 3),
        None,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "action must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        None,  #
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "reward must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        None,  #
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "pscore must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        None,  #
        "estimated_rewards_by_reg_model must be 3D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=float),  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "action elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) - 1,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "action elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        "4",  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "action must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros((3, 2), dtype=int),  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "action must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) + 8,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        r"action elements must be smaller than`",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        "4",  #
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "reward must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros((3, 2), dtype=int),  #
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "reward must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(4, dtype=int),  #
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "Expected `action.shape[0]",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        "4",  #
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "pscore must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones((5, 3)),  #
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "pscore must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(4),  #
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "Expected `action.shape[0]",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.arange(5),  #
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "pscore must be positive",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 2)),  #
        "Expected `estimated_rewards_by_reg_model.shape == action_dist.shape`, but found it False",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        "4",  #
        "estimated_rewards_by_reg_model must be 3D array",
    ),
]


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, description",
    invalid_input_of_dr,
)
def test_dr_using_invalid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    estimated_rewards_by_reg_model: np.ndarray,
    description: str,
) -> None:
    # estimate_intervals function raises ValueError of all estimators
    for estimator in [dr, sndr, dr_tuning]:
        with pytest.raises(ValueError, match=f"{description}*"):
            _ = estimator.estimate_policy_value(
                action_dist=action_dist,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )
        with pytest.raises(ValueError, match=f"{description}*"):
            _ = estimator.estimate_interval(
                action_dist=action_dist,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )


# dr variants
valid_input_of_dr_variants = [
    (
        generate_action_dist(5, 4, 3),
        np.random.choice(4, size=5),
        np.zeros(5, dtype=int),
        np.random.uniform(low=0.5, high=1.0, size=5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        0.5,
        "all arguments are given and len_list > 1",
    )
]


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, hyperparameter, description",
    valid_input_of_dr_variants,
)
def test_dr_variants_using_valid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    estimated_rewards_by_reg_model: np.ndarray,
    hyperparameter: float,
    description: str,
) -> None:
    # check dr variants
    switch_dr = SwitchDoublyRobust(lambda_=hyperparameter)
    switch_dr_tuning = SwitchDoublyRobustTuning(
        lambdas=[hyperparameter, hyperparameter * 10]
    )
    dr_os = DoublyRobustWithShrinkage(lambda_=hyperparameter)
    dr_os_tuning = DoublyRobustWithShrinkageTuning(
        lambdas=[hyperparameter, hyperparameter * 10]
    )
    for estimator in [switch_dr, switch_dr_tuning, dr_os, dr_os_tuning]:
        est = estimator.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        assert est == 0.0, f"policy value must be 0, but {est}"


def test_dr_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the format of dr variants using synthetic bandit data and random evaluation policy
    """
    expected_reward = synthetic_bandit_feedback["expected_reward"][:, :, np.newaxis]
    action_dist = random_action_dist
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_dict["action_dist"] = action_dist
    input_dict["estimated_rewards_by_reg_model"] = expected_reward
    # dr estimators require all arguments
    for estimator in dr_estimators:
        estimated_policy_value = estimator.estimate_policy_value(**input_dict)
        assert isinstance(
            estimated_policy_value, float
        ), f"invalid type response: {estimator}"
    # remove necessary keys
    del input_dict["reward"]
    del input_dict["pscore"]
    del input_dict["action"]
    del input_dict["estimated_rewards_by_reg_model"]
    for estimator in dr_estimators:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "estimate_policy_value() missing 4 required positional arguments: 'reward', 'action', 'pscore', and 'estimated_rewards_by_reg_model'"
            ),
        ):
            _ = estimator.estimate_policy_value(**input_dict)


def test_boundedness_of_sndr_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the boundedness of sndr estimators using synthetic bandit data and random evaluation policy
    """
    expected_reward = synthetic_bandit_feedback["expected_reward"][:, :, np.newaxis]
    action_dist = random_action_dist
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_dict["action_dist"] = action_dist
    input_dict["estimated_rewards_by_reg_model"] = expected_reward
    # make pscore too small (to check the boundedness of sndr)
    input_dict["pscore"] = input_dict["pscore"] ** 3
    estimated_policy_value = sndr.estimate_policy_value(**input_dict)
    assert (
        estimated_policy_value <= 2
    ), f"estimated policy value of sndr should be smaller than or equal to 2 (because of its 2-boundedness), but the value is: {estimated_policy_value}"


def test_dr_osage_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the dr shrinkage estimators using synthetic bandit data and random evaluation policy
    """
    expected_reward = synthetic_bandit_feedback["expected_reward"][:, :, np.newaxis]
    action_dist = random_action_dist
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_dict["action_dist"] = action_dist
    input_dict["estimated_rewards_by_reg_model"] = expected_reward
    dm_value = dm.estimate_policy_value(**input_dict)
    dr_value = dr.estimate_policy_value(**input_dict)
    dr_os_0_value = dr_os_0.estimate_policy_value(**input_dict)
    dr_os_max_value = dr_os_max.estimate_policy_value(**input_dict)
    assert (
        dm_value == dr_os_0_value
    ), "DoublyRobustWithShrinkage (lambda=0) should be the same as DirectMethod"
    assert (
        np.abs(dr_value - dr_os_max_value) < 1e-5
    ), "DoublyRobustWithShrinkage (lambda=inf) should be almost the same as DoublyRobust"


def test_switch_dr_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the switch_dr using synthetic bandit data and random evaluation policy
    """
    expected_reward = synthetic_bandit_feedback["expected_reward"][:, :, np.newaxis]
    action_dist = random_action_dist
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "position"]
    }
    input_dict["action_dist"] = action_dist
    input_dict["estimated_rewards_by_reg_model"] = expected_reward
    dm_value = dm.estimate_policy_value(**input_dict)
    dr_value = dr.estimate_policy_value(**input_dict)
    switch_dr_0_value = switch_dr_0.estimate_policy_value(**input_dict)
    switch_dr_max_value = switch_dr_max.estimate_policy_value(**input_dict)
    assert (
        dm_value == switch_dr_0_value
    ), "SwitchDR (lambda=0) should be the same as DirectMethod"
    assert (
        dr_value == switch_dr_max_value
    ), "SwitchDR (lambda=1e10) should be the same as DoublyRobust"
