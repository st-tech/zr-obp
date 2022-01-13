import re

from conftest import generate_action_dist
import numpy as np
import pytest

from obp.ope import InverseProbabilityWeighting
from obp.ope import InverseProbabilityWeightingTuning
from obp.ope import SelfNormalizedInverseProbabilityWeighting
from obp.ope import SubGaussianInverseProbabilityWeighting
from obp.ope import SubGaussianInverseProbabilityWeightingTuning
from obp.types import BanditFeedback


# lambda_, use_estimated_pscore, err, description
invalid_input_of_ipw_init = [
    (
        "",
        False,
        TypeError,
        r"lambda_ must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        None,
        False,
        TypeError,
        r"lambda_ must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (-1.0, False, ValueError, "lambda_ == -1.0, must be >= 0.0."),
    (np.nan, False, ValueError, "`lambda_` must not be nan"),
    (
        1.0,
        "s",
        TypeError,
        r"`use_estimated_pscore` must be a bool, but <class 'str'> is given.",
    ),
]


@pytest.mark.parametrize(
    "lambda_, use_estimated_pscore, err, description",
    invalid_input_of_ipw_init,
)
def test_ipw_init_using_invalid_inputs(
    lambda_,
    use_estimated_pscore,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = InverseProbabilityWeighting(
            lambda_=lambda_, use_estimated_pscore=use_estimated_pscore
        )


# lambdas, use_bias_upper_bound, delta, use_estimated_pscore, err, description
invalid_input_of_ipw_tuning_init = [
    (
        "",  #
        "mse",
        True,
        0.05,
        False,
        TypeError,
        "lambdas must be a list",
    ),
    (
        None,  #
        "slope",
        True,
        0.05,
        False,
        TypeError,
        "lambdas must be a list",
    ),
    (
        [""],  #
        "mse",
        True,
        0.05,
        False,
        TypeError,
        r"an element of lambdas must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        [None],  #
        "slope",
        True,
        0.05,
        False,
        TypeError,
        r"an element of lambdas must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (
        [],  #
        "mse",
        True,
        0.05,
        False,
        ValueError,
        "lambdas must not be empty",
    ),
    (
        [-1.0],  #
        "slope",
        True,
        0.05,
        False,
        ValueError,
        "an element of lambdas == -1.0, must be >= 0.0.",
    ),
    (
        [np.nan],
        "mse",
        True,
        0.05,
        False,
        ValueError,
        "an element of lambdas must not be nan",
    ),
    (
        [1],
        "",  #
        True,
        0.05,
        False,
        ValueError,
        "`tuning_method` must be either 'slope' or 'mse'",
    ),
    (
        [1],
        "mse",
        "",  #
        0.05,
        False,
        TypeError,
        "`use_bias_upper_bound` must be a bool",
    ),
    (
        [1],
        "slope",
        None,  #
        0.05,
        False,
        TypeError,
        "`use_bias_upper_bound` must be a bool",
    ),
    (
        [1],
        "mse",
        True,
        "",  #
        False,
        TypeError,
        "delta must be an instance of <class 'float'>",
    ),
    (
        [1],
        "slope",
        True,
        None,  #
        False,
        TypeError,
        "delta must be an instance of <class 'float'>",
    ),
    (
        [1],
        "mse",
        True,
        -1.0,  #
        False,
        ValueError,
        "delta == -1.0, must be >= 0.0.",
    ),
    (
        [1],
        "slope",
        True,
        1.1,  #
        False,
        ValueError,
        "delta == 1.1, must be <= 1.0.",
    ),
    (
        [1],
        "slope",
        True,
        1.0,
        "s",  #
        TypeError,
        r"`use_estimated_pscore` must be a bool, but <class 'str'> is given.",
    ),
]


@pytest.mark.parametrize(
    "lambdas, tuning_method, use_bias_upper_bound, delta, use_estimated_pscore, err, description",
    invalid_input_of_ipw_tuning_init,
)
def test_ipw_tuning_init_using_invalid_inputs(
    lambdas,
    tuning_method,
    use_bias_upper_bound,
    delta,
    use_estimated_pscore,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = InverseProbabilityWeightingTuning(
            use_bias_upper_bound=use_bias_upper_bound,
            delta=delta,
            lambdas=lambdas,
            tuning_method=tuning_method,
            use_estimated_pscore=use_estimated_pscore,
        )


# prepare ipw instances
ipw = InverseProbabilityWeighting()
snipw = SelfNormalizedInverseProbabilityWeighting()
ipw_tuning_mse = InverseProbabilityWeightingTuning(
    lambdas=[10, 1000], tuning_method="mse"
)
ipw_tuning_slope = InverseProbabilityWeightingTuning(
    lambdas=[10, 1000], tuning_method="slope"
)
sgipw_tuning_mse = SubGaussianInverseProbabilityWeightingTuning(
    lambdas=[0.01, 0.1], tuning_method="mse"
)
sgipw_tuning_slope = SubGaussianInverseProbabilityWeightingTuning(
    lambdas=[0.01, 0.1], tuning_method="slope"
)


# action_dist, action, reward, pscore, position, use_estimated_pscore, estimated_pscore, description
invalid_input_of_ipw = [
    (
        generate_action_dist(5, 4, 3),
        None,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        False,
        None,
        "`action` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        None,  #
        np.ones(5),
        np.random.choice(3, size=5),
        False,
        None,
        "`reward` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        None,  #
        np.random.choice(3, size=5),
        False,
        None,
        "`pscore` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=float),  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        False,
        None,
        "`action` elements must be integers in the range of",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) - 1,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        False,
        None,
        "`action` elements must be integers in the range of",
    ),
    (
        generate_action_dist(5, 4, 3),
        "4",  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        False,
        None,
        "`action` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros((3, 2), dtype=int),  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        False,
        None,
        "`action` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int) + 8,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        False,
        None,
        r"`action` elements must be integers in the range of`",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        "4",  #
        np.ones(5),
        np.random.choice(3, size=5),
        False,
        None,
        "`reward` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros((3, 2), dtype=int),  #
        np.ones(5),
        np.random.choice(3, size=5),
        False,
        None,
        "`reward` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(4, dtype=int),  #
        np.ones(5),
        np.random.choice(3, size=5),
        False,
        None,
        "Expected `action.shape[0]",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        "4",  #
        np.random.choice(3, size=5),
        False,
        None,
        "`pscore` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones((5, 3)),  #
        np.random.choice(3, size=5),
        False,
        None,
        "`pscore` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(4),  #
        np.random.choice(3, size=5),
        False,
        None,
        "Expected `action.shape[0]",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.arange(5),  #
        np.random.choice(3, size=5),
        False,
        None,
        "`pscore` must be positive",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        True,
        None,  #
        "`estimated_pscore` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        None,
        np.random.choice(3, size=5),
        True,
        np.arange(5),  #
        "`pscore` must be positive",
    ),
]


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, use_estimated_pscore, estimated_pscore, description",
    invalid_input_of_ipw,
)
def test_ipw_using_invalid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    use_estimated_pscore: bool,
    estimated_pscore: np.ndarray,
    description: str,
) -> None:
    # prepare ipw instances
    ipw = InverseProbabilityWeighting(use_estimated_pscore=use_estimated_pscore)
    snipw = SelfNormalizedInverseProbabilityWeighting(
        use_estimated_pscore=use_estimated_pscore
    )
    sgipw = SubGaussianInverseProbabilityWeighting(
        use_estimated_pscore=use_estimated_pscore
    )
    ipw_tuning = InverseProbabilityWeightingTuning(
        lambdas=[10, 1000], use_estimated_pscore=use_estimated_pscore
    )
    sgipw_tuning = SubGaussianInverseProbabilityWeightingTuning(
        lambdas=[0.01, 0.1], use_estimated_pscore=use_estimated_pscore
    )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = snipw.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = snipw.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw_tuning.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw_tuning.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = sgipw.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = sgipw.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = sgipw_tuning.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = sgipw_tuning.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
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
    for estimator in [ipw, snipw, ipw_tuning_mse, ipw_tuning_slope]:
        estimated_policy_value = estimator.estimate_policy_value(**input_dict)
        assert isinstance(
            estimated_policy_value, float
        ), f"invalid type response: {estimator}"

    # ipw with estimated pscore
    ipw_estimated_pscore = InverseProbabilityWeighting(use_estimated_pscore=True)
    snipw_estimated_pscore = SelfNormalizedInverseProbabilityWeighting(
        use_estimated_pscore=True
    )
    ipw_tuning_estimated_pscore = InverseProbabilityWeightingTuning(
        lambdas=[10, 1000], use_estimated_pscore=True
    )
    input_dict["estimated_pscore"] = input_dict["pscore"]
    del input_dict["pscore"]
    for estimator in [
        ipw_estimated_pscore,
        snipw_estimated_pscore,
        ipw_tuning_estimated_pscore,
    ]:
        estimated_policy_value = estimator.estimate_policy_value(**input_dict)
        assert isinstance(
            estimated_policy_value, float
        ), f"invalid type response: {estimator}"

    # remove necessary keys
    del input_dict["reward"]
    del input_dict["action"]
    for estimator in [ipw, snipw]:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "estimate_policy_value() missing 2 required positional arguments: 'reward' and 'action'"
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

    # ipw with estimated pscore
    snipw_estimated_pscore = SelfNormalizedInverseProbabilityWeighting(
        use_estimated_pscore=True
    )
    input_dict["estimated_pscore"] = input_dict["pscore"]
    del input_dict["pscore"]
    estimated_policy_value = snipw_estimated_pscore.estimate_policy_value(**input_dict)
    assert (
        estimated_policy_value <= 1
    ), f"estimated policy value of snipw should be smaller than or equal to 1 (because of its 1-boundedness), but the value is: {estimated_policy_value}"
