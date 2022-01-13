import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from obp.dataset import linear_behavior_policy_logit
from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticSlateBanditDataset
from obp.ope import SlateCascadeDoublyRobust
from obp.ope import SlateRegressionModel
from obp.ope import SlateRewardInteractionIPS


# setting
len_list = 3
n_unique_action = 10
rips = SlateRewardInteractionIPS(len_list=len_list)
dr = SlateCascadeDoublyRobust(len_list=len_list, n_unique_action=n_unique_action)
n_rounds = 5

# --- invalid ---
# slate_id, action, reward, pscore, position, evaluation_policy_pscore, q_hat, evaluation_policy_action_dist, description
invalid_input_of_slate_estimators = [
    (
        "4",  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`slate_id` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list).reshape((n_rounds, len_list)),  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`slate_id` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list) - 1,  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "slate_id elements must be non-negative integers",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        "4",  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`action` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros((n_rounds, len_list), dtype=int),  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`action` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int) - 1,  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`action` elements must be integers in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=float),  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`action` elements must be integers in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.ones(n_rounds * len_list, dtype=int) * 10,  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`action` elements must be integers in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        "4",  #
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`reward` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros((n_rounds, len_list), dtype=int),  #
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`reward` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros((n_rounds * len_list), dtype=int),
        "4",  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`pscore_cascade` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros((n_rounds * len_list), dtype=int),
        np.ones((n_rounds, len_list)),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`pscore_cascade` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros((n_rounds * len_list), dtype=int),
        np.ones(n_rounds * len_list) + 1,  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`pscore_cascade` must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros((n_rounds * len_list), dtype=int),
        np.ones(n_rounds * len_list) - 1,  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`pscore_cascade` must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros((n_rounds * len_list), dtype=int),
        np.hstack([[0.2], np.ones(n_rounds * len_list - 1)]),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`pscore_cascade` must be non-increasing sequence in each slate",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros((n_rounds * len_list), dtype=int),
        np.ones(n_rounds * len_list - 1),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`slate_id`, `position`, `reward`, `pscore_cascade`, and `evaluation_policy_pscore_cascade` must have the same number of samples",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        "4",  #
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`position` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds).reshape((n_rounds, len_list)),  #
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`position` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds) - 1,  #
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`position` elements must be non-negative integers",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.repeat(np.arange(n_rounds), len_list),  #
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`position` must not be duplicated in each slate",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        "4",  #
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`evaluation_policy_pscore_cascade` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones((n_rounds, len_list)),  #
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`evaluation_policy_pscore_cascade` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list) + 1,  #
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`evaluation_policy_pscore_cascade` must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list) - 1.1,  #
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`evaluation_policy_pscore_cascade` must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.hstack([[0.2], np.ones(n_rounds * len_list - 1)]),  #
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`evaluation_policy_pscore_cascade` must be non-increasing sequence in each slate",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        None,  #
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`q_hat` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "4",  #
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`q_hat` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones((n_rounds, len_list, n_unique_action)),  #
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`q_hat` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones((n_rounds * len_list, n_unique_action)),  #
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "`q_hat` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        "4",  #
        "`evaluation_policy_action_dist` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones((n_rounds, len_list, n_unique_action)) / n_unique_action,  #
        "`evaluation_policy_action_dist` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones((n_rounds * len_list, n_unique_action)) / n_unique_action,  #
        "`evaluation_policy_action_dist` must be 1D array",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action),  #
        "evaluation_policy_action_dist[i * n_unique_action : (i+1) * n_unique_action]",
    ),
]


@pytest.mark.parametrize(
    "slate_id, action, reward, pscore, position, evaluation_policy_pscore, q_hat, evaluation_policy_action_dist, description",
    invalid_input_of_slate_estimators,
)
def test_estimate_policy_value_using_invalid_input_data(
    slate_id,
    action,
    reward,
    pscore,
    position,
    evaluation_policy_pscore,
    q_hat,
    evaluation_policy_action_dist,
    description,
) -> None:
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = dr.estimate_policy_value(
            slate_id=slate_id,
            action=action,
            reward=reward,
            pscore_cascade=pscore,
            position=position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore,
            q_hat=q_hat,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )
        _ = dr.estimate_interval(
            slate_id=slate_id,
            action=action,
            reward=reward,
            pscore_cascade=pscore,
            position=position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore,
            q_hat=q_hat,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )


# --- valid ---
valid_input_of_slate_estimators = [
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action,
        "each slate has data of 3 (len_list) positions",
    ),
]


@pytest.mark.parametrize(
    "slate_id, action, reward, pscore, position, evaluation_policy_pscore, q_hat, evaluation_policy_action_dist, description",
    valid_input_of_slate_estimators,
)
def test_cascade_dr_using_valid_input_data(
    slate_id,
    action,
    reward,
    pscore,
    position,
    evaluation_policy_pscore,
    q_hat,
    evaluation_policy_action_dist,
    description,
) -> None:
    _ = dr.estimate_policy_value(
        slate_id=slate_id,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        q_hat=q_hat,
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )
    _ = dr.estimate_interval(
        slate_id=slate_id,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        q_hat=q_hat,
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )


# --- confidence intervals ---
# alpha, n_bootstrap_samples, random_state, err, description
invalid_input_of_estimate_intervals = [
    (
        0.05,
        100,
        "s",
        ValueError,
        "'s' cannot be used to seed a numpy.random.RandomState instance",
    ),
    (0.05, -1, 1, ValueError, "n_bootstrap_samples == -1, must be >= 1"),
    (
        0.05,
        "s",
        1,
        TypeError,
        "n_bootstrap_samples must be an instance of <class 'int'>, not <class 'str'>",
    ),
    (-1.0, 1, 1, ValueError, "alpha == -1.0, must be >= 0.0"),
    (2.0, 1, 1, ValueError, "alpha == 2.0, must be <= 1.0"),
    (
        "0",
        1,
        1,
        TypeError,
        "alpha must be an instance of <class 'float'>, not <class 'str'>",
    ),
]

valid_input_of_estimate_intervals = [
    (0.05, 100, 1, "random_state is 1"),
    (0.05, 1, 1, "n_bootstrap_samples is 1"),
]


@pytest.mark.parametrize(
    "slate_id, action, reward, pscore, position, evaluation_policy_pscore, q_hat, evaluation_policy_action_dist, description_1",
    valid_input_of_slate_estimators,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, err, description_2",
    invalid_input_of_estimate_intervals,
)
def test_estimate_interval_using_invalid_input_data(
    slate_id,
    action,
    reward,
    pscore,
    position,
    evaluation_policy_pscore,
    q_hat,
    evaluation_policy_action_dist,
    description_1,
    alpha,
    n_bootstrap_samples,
    random_state,
    err,
    description_2,
) -> None:
    with pytest.raises(err, match=f"{description_2}*"):
        _ = dr.estimate_interval(
            slate_id=slate_id,
            action=action,
            reward=reward,
            pscore_cascade=pscore,
            position=position,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore,
            q_hat=q_hat,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "slate_id, action, reward, pscore, position, evaluation_policy_pscore, q_hat, evaluation_policy_action_dist, description_1",
    valid_input_of_slate_estimators,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    valid_input_of_estimate_intervals,
)
def test_estimate_interval_using_valid_input_data(
    slate_id,
    action,
    reward,
    pscore,
    position,
    evaluation_policy_pscore,
    q_hat,
    evaluation_policy_action_dist,
    description_1,
    alpha,
    n_bootstrap_samples,
    random_state,
    description_2,
) -> None:
    _ = dr.estimate_interval(
        slate_id=slate_id,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        q_hat=q_hat,
        evaluation_policy_action_dist=evaluation_policy_action_dist,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )


def test_slate_ope_performance_using_cascade_additive_log():
    # set parameters
    n_unique_action = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 1000
    reward_structure = "cascade_additive"
    click_model = None
    behavior_policy_function = linear_behavior_policy_logit
    reward_function = logistic_reward_function
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=behavior_policy_function,
        base_reward_function=reward_function,
    )
    random_behavior_dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=None,
        base_reward_function=reward_function,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    slate_id = bandit_feedback["slate_id"]
    context = bandit_feedback["context"]
    action = bandit_feedback["action"]
    reward = bandit_feedback["reward"]
    pscore = bandit_feedback["pscore_cascade"]
    position = bandit_feedback["position"]

    # obtain random behavior feedback
    random_behavior_feedback = random_behavior_dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds
    )
    evaluation_policy_logit_ = np.ones((n_rounds, n_unique_action)) / n_unique_action
    evaluation_policy_action_dist = (
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action
    )
    (
        _,
        _,
        evaluation_policy_pscore,
    ) = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
        return_pscore_item_position=False,
    )
    evaluation_policy_action_dist = dataset.calc_evaluation_policy_action_dist(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
    )

    # obtain q_hat
    base_regression_model = SlateRegressionModel(
        base_model=DecisionTreeRegressor(max_depth=3, random_state=12345),
        len_list=len_list,
        n_unique_action=n_unique_action,
        fitting_method="iw",
    )
    q_hat = base_regression_model.fit_predict(
        context=context,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )

    # check if q_hat=0 case coincides with rips
    cascade_dr_estimated_policy_value = dr.estimate_policy_value(
        slate_id=slate_id,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        q_hat=q_hat,
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )
    # compute statistics of ground truth policy value
    q_pi_e = (
        random_behavior_feedback["reward"]
        .reshape((n_rounds, dataset.len_list))
        .sum(axis=1)
    )
    gt_mean = q_pi_e.mean()
    gt_std = q_pi_e.std(ddof=1)
    print("Cascade additive")
    # check the performance of OPE
    ci_bound = gt_std * 3 / np.sqrt(q_pi_e.shape[0])
    print(f"gt_mean: {gt_mean}, 3 * gt_std / sqrt(n): {ci_bound}")
    estimated_policy_value = {
        "cascade-dr": cascade_dr_estimated_policy_value,
    }
    for key in estimated_policy_value:
        print(
            f"estimated_value: {estimated_policy_value[key]} ------ estimator: {key}, "
        )
        # test the performance of each estimator
        assert (
            np.abs(gt_mean - estimated_policy_value[key]) <= ci_bound
        ), f"OPE of {key} did not work well (absolute error is greater than 3*sigma)"

    # check if q_hat = 0 case of cascade-dr coincides with rips
    cascade_dr_estimated_policy_value_ = dr.estimate_policy_value(
        slate_id=slate_id,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        q_hat=np.zeros_like(q_hat),
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )
    rips_estimated_policy_value = rips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
    )
    assert np.allclose(
        np.array([cascade_dr_estimated_policy_value_]),
        np.array([rips_estimated_policy_value]),
    )


def test_slate_ope_performance_using_independent_log():
    # set parameters
    n_unique_action = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 1000
    reward_structure = "independent"
    click_model = None
    behavior_policy_function = linear_behavior_policy_logit
    reward_function = logistic_reward_function
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=behavior_policy_function,
        base_reward_function=reward_function,
    )
    random_behavior_dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=None,
        base_reward_function=reward_function,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    slate_id = bandit_feedback["slate_id"]
    context = bandit_feedback["context"]
    action = bandit_feedback["action"]
    reward = bandit_feedback["reward"]
    pscore = bandit_feedback["pscore_cascade"]
    position = bandit_feedback["position"]

    # obtain random behavior feedback
    random_behavior_feedback = random_behavior_dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds
    )
    evaluation_policy_logit_ = np.ones((n_rounds, n_unique_action)) / n_unique_action
    evaluation_policy_action_dist = (
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action
    )
    (
        _,
        _,
        evaluation_policy_pscore,
    ) = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
        return_pscore_item_position=False,
    )
    evaluation_policy_action_dist = dataset.calc_evaluation_policy_action_dist(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
    )

    # obtain q_hat
    base_regression_model = SlateRegressionModel(
        base_model=DecisionTreeRegressor(max_depth=3, random_state=12345),
        len_list=len_list,
        n_unique_action=n_unique_action,
        fitting_method="iw",
    )
    q_hat = base_regression_model.fit_predict(
        context=context,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )

    # check if q_hat=0 case coincides with rips
    cascade_dr_estimated_policy_value = dr.estimate_policy_value(
        slate_id=slate_id,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        q_hat=q_hat,
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )
    # compute statistics of ground truth policy value
    q_pi_e = (
        random_behavior_feedback["reward"]
        .reshape((n_rounds, dataset.len_list))
        .sum(axis=1)
    )
    gt_mean = q_pi_e.mean()
    gt_std = q_pi_e.std(ddof=1)
    print("Cascade additive")
    # check the performance of OPE
    ci_bound = gt_std * 3 / np.sqrt(q_pi_e.shape[0])
    print(f"gt_mean: {gt_mean}, 3 * gt_std / sqrt(n): {ci_bound}")
    estimated_policy_value = {
        "cascade-dr": cascade_dr_estimated_policy_value,
    }
    for key in estimated_policy_value:
        print(
            f"estimated_value: {estimated_policy_value[key]} ------ estimator: {key}, "
        )
        # test the performance of each estimator
        assert (
            np.abs(gt_mean - estimated_policy_value[key]) <= ci_bound
        ), f"OPE of {key} did not work well (absolute error is greater than 3*sigma)"

    # check if q_hat = 0 case of cascade-dr coincides with rips
    cascade_dr_estimated_policy_value_ = dr.estimate_policy_value(
        slate_id=slate_id,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        q_hat=np.zeros_like(q_hat),
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )
    rips_estimated_policy_value = rips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
    )
    assert np.allclose(
        np.array([cascade_dr_estimated_policy_value_]),
        np.array([rips_estimated_policy_value]),
    )


def test_slate_ope_performance_using_standard_additive_log():
    # set parameters
    n_unique_action = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 1000
    reward_structure = "standard_additive"
    click_model = None
    behavior_policy_function = linear_behavior_policy_logit
    reward_function = logistic_reward_function
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=behavior_policy_function,
        base_reward_function=reward_function,
    )
    random_behavior_dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=None,
        base_reward_function=reward_function,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    slate_id = bandit_feedback["slate_id"]
    context = bandit_feedback["context"]
    action = bandit_feedback["action"]
    reward = bandit_feedback["reward"]
    pscore = bandit_feedback["pscore_cascade"]
    position = bandit_feedback["position"]

    # obtain random behavior feedback
    random_behavior_feedback = random_behavior_dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds
    )
    evaluation_policy_logit_ = np.ones((n_rounds, n_unique_action)) / n_unique_action
    evaluation_policy_action_dist = (
        np.ones(n_rounds * len_list * n_unique_action) / n_unique_action
    )
    (
        _,
        _,
        evaluation_policy_pscore,
    ) = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
        return_pscore_item_position=False,
    )
    evaluation_policy_action_dist = dataset.calc_evaluation_policy_action_dist(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
    )

    # obtain q_hat
    base_regression_model = SlateRegressionModel(
        base_model=DecisionTreeRegressor(max_depth=3, random_state=12345),
        len_list=len_list,
        n_unique_action=n_unique_action,
        fitting_method="iw",
    )
    q_hat = base_regression_model.fit_predict(
        context=context,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )

    # check if q_hat=0 case coincides with rips
    cascade_dr_estimated_policy_value = dr.estimate_policy_value(
        slate_id=slate_id,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        q_hat=q_hat,
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )
    # compute statistics of ground truth policy value
    q_pi_e = (
        random_behavior_feedback["reward"]
        .reshape((n_rounds, dataset.len_list))
        .sum(axis=1)
    )
    gt_mean = q_pi_e.mean()
    gt_std = q_pi_e.std(ddof=1)
    print("Cascade additive")
    # check the performance of OPE
    ci_bound = gt_std * 3 / np.sqrt(q_pi_e.shape[0])
    print(f"gt_mean: {gt_mean}, 3 * gt_std / sqrt(n): {ci_bound}")
    estimated_policy_value = {
        "cascade-dr": cascade_dr_estimated_policy_value,
    }
    for key in estimated_policy_value:
        print(
            f"estimated_value: {estimated_policy_value[key]} ------ estimator: {key}, "
        )
        # test the performance of each estimator
        assert (
            np.abs(gt_mean - estimated_policy_value[key]) <= ci_bound
        ), f"OPE of {key} did not work well (absolute error is greater than 3*sigma)"

    # check if q_hat = 0 case of cascade-dr coincides with rips
    cascade_dr_estimated_policy_value_ = dr.estimate_policy_value(
        slate_id=slate_id,
        action=action,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
        q_hat=np.zeros_like(q_hat),
        evaluation_policy_action_dist=evaluation_policy_action_dist,
    )
    rips_estimated_policy_value = rips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore_cascade=evaluation_policy_pscore,
    )
    assert np.allclose(
        np.array([cascade_dr_estimated_policy_value_]),
        np.array([rips_estimated_policy_value]),
    )
