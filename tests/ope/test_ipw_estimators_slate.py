import pytest
import numpy as np

from obp.ope import SlateStandardIPS, SlateIndependentIPS, SlateRecursiveIPS
from obp.dataset import (
    logistic_reward_function,
    linear_behavior_policy_logit,
    SyntheticSlateBanditDataset,
)

# setting
len_list = 3
sips = SlateStandardIPS(len_list=len_list)
iips = SlateIndependentIPS(len_list=len_list)
rips = SlateRecursiveIPS(len_list=len_list)
n_rounds = 5


# --- _extract_reward_by_bootstrap ---
def test_extract_reward_by_bootstrap() -> None:
    slate_id = np.repeat(np.arange(5), 3)
    estimated_rewards = np.random.normal(size=n_rounds * len_list)
    sampled_slate = np.array([0, 3, 0])
    sampled_estimated_rewards = sips._extract_reward_by_bootstrap(
        slate_id=slate_id,
        estimated_rewards=estimated_rewards,
        sampled_slate=sampled_slate,
    )
    correct_sampled_estimated_rewards = np.hstack(
        [estimated_rewards[0:3], estimated_rewards[9:12], estimated_rewards[0:3]]
    )
    assert np.allclose(sampled_estimated_rewards, correct_sampled_estimated_rewards)


# --- invalid (all slate estimators) ---

# slate_id, reward, pscore, position, evaluation_policy_pscore, description
invalid_input_of_slate_estimators = [
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        "4",  #
        np.ones(n_rounds * len_list),
        "position must be ndarray",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds).reshape((n_rounds, len_list)),  #
        np.ones(n_rounds * len_list),
        "position must be 1-dimensional",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds) - 1,  #
        np.ones(n_rounds * len_list),
        "position elements must be non-negative integers",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        "4",  #
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "reward must be ndarray",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros((n_rounds, len_list), dtype=int),  #
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "reward must be 1-dimensional",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        "4",  #
        "evaluation_policy_pscore must be ndarray",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones((n_rounds, len_list)),  #
        "evaluation_policy_pscore must be 1-dimensional",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list) + 1,  #
        "evaluation_policy_pscore must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list) - 1,  #
        "evaluation_policy_pscore must be in the range of",
    ),
    (
        "4",  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "slate_id must be ndarray",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list).reshape((n_rounds, len_list)),  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "slate_id must be 1-dimensional",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list) - 1,  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "slate_id elements must be non-negative integers",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),  #
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.repeat(np.arange(n_rounds), len_list),  #
        np.ones(n_rounds * len_list),
        "position must not be duplicated in each slate",
    ),
]


@pytest.mark.parametrize(
    "slate_id, reward, pscore, position, evaluation_policy_pscore, description",
    invalid_input_of_slate_estimators,
)
def test_slate_estimators_using_invalid_input_data(
    slate_id, reward, pscore, position, evaluation_policy_pscore, description
) -> None:
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = sips.estimate_policy_value(
            slate_id=slate_id,
            reward=reward,
            pscore=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        _ = sips.estimate_interval(
            slate_id=slate_id,
            reward=reward,
            pscore=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        _ = iips.estimate_policy_value(
            slate_id=slate_id,
            reward=reward,
            pscore_item_position=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        _ = iips.estimate_interval(
            slate_id=slate_id,
            reward=reward,
            pscore_item_position=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        _ = rips.estimate_policy_value(
            slate_id=slate_id,
            reward=reward,
            pscore_cascade=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        _ = rips.estimate_interval(
            slate_id=slate_id,
            reward=reward,
            pscore_cascade=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )


# --- valid (all slate estimators) ---

valid_input_of_slate_estimators = [
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list),
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "each slate has data of 3 (len_list) positions",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list)[:-1],
        np.zeros(n_rounds * len_list, dtype=int)[:-1],
        np.ones(n_rounds * len_list)[:-1],
        np.tile(np.arange(len_list), n_rounds)[:-1],
        np.ones(n_rounds * len_list)[:-1],
        "last slate has data of 2 (len_list - 1) positions",
    ),
]


@pytest.mark.parametrize(
    "slate_id, reward, pscore, position, evaluation_policy_pscore, description",
    valid_input_of_slate_estimators,
)
def test_slate_estimators_using_valid_input_data(
    slate_id, reward, pscore, position, evaluation_policy_pscore, description
) -> None:
    _ = sips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore=pscore,
        position=position,
        evaluation_policy_pscore=evaluation_policy_pscore,
    )
    _ = sips.estimate_interval(
        slate_id=slate_id,
        reward=reward,
        pscore=pscore,
        position=position,
        evaluation_policy_pscore=evaluation_policy_pscore,
    )
    _ = iips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore_item_position=pscore,
        position=position,
        evaluation_policy_pscore=evaluation_policy_pscore,
    )
    _ = iips.estimate_interval(
        slate_id=slate_id,
        reward=reward,
        pscore_item_position=pscore,
        position=position,
        evaluation_policy_pscore=evaluation_policy_pscore,
    )
    _ = rips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore=evaluation_policy_pscore,
    )
    _ = rips.estimate_interval(
        slate_id=slate_id,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore=evaluation_policy_pscore,
    )


# --- invalid (sips) ---
invalid_input_of_sips = [
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        "4",  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore must be ndarray",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones((n_rounds, len_list)),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore must be 1-dimensional",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list) + 1,  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list) - 1,  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list - 1),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "slate_id, position, reward, pscore, and evaluation_policy_pscore must be the same size",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.hstack([np.ones(n_rounds * len_list - 1), [0.2]]),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore must be unique in each slate",
    ),
]


@pytest.mark.parametrize(
    "slate_id, reward, pscore, position, evaluation_policy_pscore, description",
    invalid_input_of_sips,
)
def test_sips_using_invalid_input_data(
    slate_id, reward, pscore, position, evaluation_policy_pscore, description
) -> None:
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = sips.estimate_policy_value(
            slate_id=slate_id,
            reward=reward,
            pscore=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        _ = sips.estimate_interval(
            slate_id=slate_id,
            reward=reward,
            pscore=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )


# --- invalid (iips) ---
invalid_input_of_iips = [
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        "4",  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore_item_position must be ndarray",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones((n_rounds, len_list)),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore_item_position must be 1-dimensional",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list) + 1,  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore_item_position must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list) - 1,  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore_item_position must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list - 1),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "slate_id, position, reward, pscore_item_position, and evaluation_policy_pscore must be the same size",
    ),
]


@pytest.mark.parametrize(
    "slate_id, reward, pscore_item_position, position, evaluation_policy_pscore, description",
    invalid_input_of_iips,
)
def test_iips_using_invalid_input_data(
    slate_id,
    reward,
    pscore_item_position,
    position,
    evaluation_policy_pscore,
    description,
) -> None:
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = iips.estimate_policy_value(
            slate_id=slate_id,
            reward=reward,
            pscore_item_position=pscore_item_position,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        _ = iips.estimate_interval(
            slate_id=slate_id,
            reward=reward,
            pscore_item_position=pscore_item_position,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )


# --- invalid (rips) ---
invalid_input_of_rips = [
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        "4",  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore_cascade must be ndarray",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones((n_rounds, len_list)),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore_cascade must be 1-dimensional",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list) + 1,  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore_cascade must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list) - 1,  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore_cascade must be in the range of",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.ones(n_rounds * len_list - 1),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "slate_id, position, reward, pscore_cascade, and evaluation_policy_pscore must be the same size",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list, dtype=int),
        np.hstack([[0.2], np.ones(n_rounds * len_list - 1)]),  #
        np.tile(np.arange(len_list), n_rounds),
        np.ones(n_rounds * len_list),
        "pscore_cascade must be non-increasing sequence in each slate",
    ),
]


@pytest.mark.parametrize(
    "slate_id, reward, pscore_cascade, position, evaluation_policy_pscore, description",
    invalid_input_of_rips,
)
def test_rips_using_invalid_input_data(
    slate_id, reward, pscore_cascade, position, evaluation_policy_pscore, description
) -> None:
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = rips.estimate_policy_value(
            slate_id=slate_id,
            reward=reward,
            pscore_cascade=pscore_cascade,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        _ = rips.estimate_interval(
            slate_id=slate_id,
            reward=reward,
            pscore_cascade=pscore_cascade,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )


# --- confidence intervals ---
# alpha, n_bootstrap_samples, random_state, description
invalid_input_of_estimate_intervals = [
    (0.05, 100, "s", "random_state must be an integer"),
    (0.05, -1, 1, "n_bootstrap_samples must be a positive integer"),
    (0.05, "s", 1, "n_bootstrap_samples must be a positive integer"),
    (0.0, 1, 1, "alpha must be a positive float (< 1)"),
    (1.0, 1, 1, "alpha must be a positive float (< 1)"),
    ("0", 1, 1, "alpha must be a positive float (< 1)"),
]

valid_input_of_estimate_intervals = [
    (0.05, 100, 1, "random_state is 1"),
    (0.05, 1, 1, "n_bootstrap_samples is 1"),
]


@pytest.mark.parametrize(
    "slate_id, reward, pscore, position, evaluation_policy_pscore, description_1",
    valid_input_of_slate_estimators,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    invalid_input_of_estimate_intervals,
)
def test_estimate_intervals_of_all_estimators_using_invalid_input_data(
    slate_id,
    reward,
    pscore,
    position,
    evaluation_policy_pscore,
    description_1,
    alpha,
    n_bootstrap_samples,
    random_state,
    description_2,
) -> None:
    with pytest.raises(ValueError, match=f"{description_2}*"):
        _ = sips.estimate_interval(
            slate_id=slate_id,
            reward=reward,
            pscore=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        _ = iips.estimate_interval(
            slate_id=slate_id,
            reward=reward,
            pscore_item_position=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        _ = rips.estimate_interval(
            slate_id=slate_id,
            reward=reward,
            pscore_cascade=pscore,
            position=position,
            evaluation_policy_pscore=evaluation_policy_pscore,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "slate_id, reward, pscore, position, evaluation_policy_pscore, description_1",
    valid_input_of_slate_estimators,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description_2",
    valid_input_of_estimate_intervals,
)
def test_estimate_intervals_of_all_estimators_using_valid_input_data(
    slate_id,
    reward,
    pscore,
    position,
    evaluation_policy_pscore,
    description_1,
    alpha,
    n_bootstrap_samples,
    random_state,
    description_2,
) -> None:
    _ = sips.estimate_interval(
        slate_id=slate_id,
        reward=reward,
        pscore=pscore,
        position=position,
        evaluation_policy_pscore=evaluation_policy_pscore,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )
    _ = iips.estimate_interval(
        slate_id=slate_id,
        reward=reward,
        pscore_item_position=pscore,
        position=position,
        evaluation_policy_pscore=evaluation_policy_pscore,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )
    _ = rips.estimate_interval(
        slate_id=slate_id,
        reward=reward,
        pscore_cascade=pscore,
        position=position,
        evaluation_policy_pscore=evaluation_policy_pscore,
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
    reward = bandit_feedback["reward"]
    pscore = bandit_feedback["pscore"]
    pscore_item_position = bandit_feedback["pscore_item_position"]
    pscore_cascade = bandit_feedback["pscore_cascade"]
    position = bandit_feedback["position"]

    # obtain random behavior feedback
    random_behavior_feedback = random_behavior_dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_exact_uniform_pscore_item_position=True
    )

    sips_estimated_policy_value = sips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore=pscore,
        position=position,
        evaluation_policy_pscore=random_behavior_feedback["pscore"],
    )
    iips_estimated_policy_value = iips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore_item_position=pscore_item_position,
        position=position,
        evaluation_policy_pscore=random_behavior_feedback["pscore_item_position"],
    )
    rips_estimated_policy_value = rips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore_cascade=pscore_cascade,
        position=position,
        evaluation_policy_pscore=random_behavior_feedback["pscore_cascade"],
    )
    # compute statistics of ground truth policy value
    q_pi_e = (
        random_behavior_feedback["reward"]
        .reshape((n_rounds, dataset.len_list))
        .mean(axis=1)
    )
    gt_mean = q_pi_e.mean()
    gt_std = q_pi_e.std(ddof=1)
    print("Cascade additive")
    # check the performance of OPE
    ci_bound = gt_std * 3 / np.sqrt(q_pi_e.shape[0])
    print(f"gt_mean: {gt_mean}, 3 * gt_std / sqrt(n): {ci_bound}")
    estimated_policy_value = {
        "sips": sips_estimated_policy_value,
        "iips": iips_estimated_policy_value,
        "rips": rips_estimated_policy_value,
    }
    for key in estimated_policy_value:
        print(
            f"estimated_value: {estimated_policy_value[key]} ------ estimator: {key}, "
        )
        # test the performance of each estimator
        assert (
            np.abs(gt_mean - estimated_policy_value[key]) <= ci_bound
        ), f"OPE of {key} did not work well (absolute error is greater than 3*sigma)"


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
    reward = bandit_feedback["reward"]
    pscore = bandit_feedback["pscore"]
    pscore_item_position = bandit_feedback["pscore_item_position"]
    pscore_cascade = bandit_feedback["pscore_cascade"]
    position = bandit_feedback["position"]

    # obtain random behavior feedback
    random_behavior_feedback = random_behavior_dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_exact_uniform_pscore_item_position=True
    )

    sips_estimated_policy_value = sips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore=pscore,
        position=position,
        evaluation_policy_pscore=random_behavior_feedback["pscore"],
    )
    iips_estimated_policy_value = iips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore_item_position=pscore_item_position,
        position=position,
        evaluation_policy_pscore=random_behavior_feedback["pscore_item_position"],
    )
    rips_estimated_policy_value = rips.estimate_policy_value(
        slate_id=slate_id,
        reward=reward,
        pscore_cascade=pscore_cascade,
        position=position,
        evaluation_policy_pscore=random_behavior_feedback["pscore_cascade"],
    )
    # compute statistics of ground truth policy value
    q_pi_e = (
        random_behavior_feedback["reward"]
        .reshape((n_rounds, dataset.len_list))
        .mean(axis=1)
    )
    gt_mean = q_pi_e.mean()
    gt_std = q_pi_e.std(ddof=1)
    print("Independent")
    # check the performance of OPE
    ci_bound = gt_std * 3 / np.sqrt(q_pi_e.shape[0])
    print(f"gt_mean: {gt_mean}, 3 * gt_std / sqrt(n): {ci_bound}")
    estimated_policy_value = {
        "sips": sips_estimated_policy_value,
        "iips": iips_estimated_policy_value,
        "rips": rips_estimated_policy_value,
    }
    for key in estimated_policy_value:
        print(
            f"estimated_value: {estimated_policy_value[key]} ------ estimator: {key}, "
        )
        # test the performance of each estimator
        assert (
            np.abs(gt_mean - estimated_policy_value[key]) <= ci_bound
        ), f"OPE of {key} did not work well (absolute error is greater than 3*sigma)"