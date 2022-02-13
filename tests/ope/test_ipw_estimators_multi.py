import re

from conftest import generate_action_dist
import numpy as np
import pytest

from obp.ope import MultiLoggersBalancedInverseProbabilityWeighting as BalIPW
from obp.ope import MultiLoggersNaiveInverseProbabilityWeighting as NaiveIPW
from obp.ope import MultiLoggersWeightedInverseProbabilityWeighting as WeightedIPW
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
        _ = NaiveIPW(lambda_=lambda_, use_estimated_pscore=use_estimated_pscore)

    with pytest.raises(err, match=f"{description}*"):
        _ = BalIPW(lambda_=lambda_, use_estimated_pscore=use_estimated_pscore)

    with pytest.raises(err, match=f"{description}*"):
        _ = WeightedIPW(lambda_=lambda_, use_estimated_pscore=use_estimated_pscore)


# action_dist, action, reward, pscore, stratum_idx, position, use_estimated_pscore, estimated_pscore, description
invalid_input_of_weighted_ipw = [
    (
        generate_action_dist(5, 4, 3),
        None,  #
        np.zeros(5, dtype=int),
        np.ones(5),
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
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
        np.ones((5, 3), dtype=int),  #
        np.random.choice(3, size=5),
        False,
        None,
        "`stratum_idx` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.arange(4, dtype=int),  #
        np.random.choice(3, size=5),
        False,
        None,
        "Expected `action_dist.shape[0]",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.arange(5, dtype=int) - 1,  #
        np.random.choice(3, size=5),
        False,
        None,
        "`stratum_idx` elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.arange(5, dtype=int),
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
        np.arange(5, dtype=int),
        np.random.choice(3, size=5),
        True,
        np.arange(5),  #
        "`pscore` must be positive",
    ),
]


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, stratum_idx, position, use_estimated_pscore, estimated_pscore, description",
    invalid_input_of_weighted_ipw,
)
def test_weighted_ipw_using_invalid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    stratum_idx: np.ndarray,
    position: np.ndarray,
    use_estimated_pscore: bool,
    estimated_pscore: np.ndarray,
    description: str,
) -> None:
    # prepare ipw instances
    weighted_ipw = WeightedIPW(use_estimated_pscore=use_estimated_pscore)
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = weighted_ipw.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            stratum_idx=stratum_idx,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = weighted_ipw.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            stratum_idx=stratum_idx,
            position=position,
            estimated_pscore=estimated_pscore,
        )


# action_dist, action, reward, pscore, position, use_estimated_pscore, estimated_pscore, description
invalid_input_of_naive_ipw = [
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
    invalid_input_of_naive_ipw,
)
def test_naive_ipw_using_invalid_input_data(
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
    naive_ipw = NaiveIPW(use_estimated_pscore=use_estimated_pscore)
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = naive_ipw.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = naive_ipw.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_pscore=estimated_pscore,
        )


# action_dist, action, reward, pscore_avg, position, use_estimated_pscore, estimated_pscore, description
invalid_input_of_bal_ipw = [
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
        "`pscore_avg` must be 1D array",
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
        "`pscore_avg` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones((5, 3)),  #
        np.random.choice(3, size=5),
        False,
        None,
        "`pscore_avg` must be 1D array",
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
        "`estimated_pscore_avg` must be 1D array",
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
    "action_dist, action, reward, pscore_avg, position, use_estimated_pscore, estimated_pscore, description",
    invalid_input_of_bal_ipw,
)
def test_bal_ipw_using_invalid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore_avg: np.ndarray,
    position: np.ndarray,
    use_estimated_pscore: bool,
    estimated_pscore: np.ndarray,
    description: str,
) -> None:
    # prepare ipw instances
    bal_ipw = BalIPW(use_estimated_pscore=use_estimated_pscore)
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = bal_ipw.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore_avg=pscore_avg,
            position=position,
            estimated_pscore_avg=estimated_pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = bal_ipw.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore_avg=pscore_avg,
            position=position,
            estimated_pscore_avg=estimated_pscore,
        )


def test_ipw_using_random_evaluation_policy(
    synthetic_multi_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the format of ipw variants using synthetic bandit data and random evaluation policy
    """
    action_dist = random_action_dist
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_multi_bandit_feedback.items()
        if k in ["reward", "action", "pscore", "pscore_avg", "stratum_idx", "position"]
    }
    input_dict["action_dist"] = action_dist
    naive_ipw = NaiveIPW()
    bal_ipw = BalIPW()
    weighted_ipw = WeightedIPW()
    # ipw estimators can be used without estimated_rewards_by_reg_model
    for estimator in [naive_ipw, bal_ipw, weighted_ipw]:
        estimated_policy_value = estimator.estimate_policy_value(**input_dict)
        assert isinstance(
            estimated_policy_value, float
        ), f"invalid type response: {estimator}"

    # ipw with estimated pscore
    naive_ipw = NaiveIPW(use_estimated_pscore=True)
    bal_ipw = BalIPW(use_estimated_pscore=True)
    weighted_ipw = WeightedIPW(use_estimated_pscore=True)
    input_dict["estimated_pscore"] = input_dict["pscore"]
    input_dict["estimated_pscore_avg"] = input_dict["pscore"]
    del input_dict["pscore"]
    del input_dict["pscore_avg"]
    for estimator in [naive_ipw, bal_ipw, weighted_ipw]:
        estimated_policy_value = estimator.estimate_policy_value(**input_dict)
        assert isinstance(
            estimated_policy_value, float
        ), f"invalid type response: {estimator}"

    # remove necessary keys
    del input_dict["reward"]
    del input_dict["action"]
    for estimator in [naive_ipw, weighted_ipw, bal_ipw]:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "estimate_policy_value() missing 2 required positional arguments: 'reward' and 'action'"
            ),
        ):
            _ = estimator.estimate_policy_value(**input_dict)
