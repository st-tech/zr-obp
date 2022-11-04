import numpy as np
import pytest
from obp.dataset.synthetic import logistic_sparse_reward_function

from obp.simulator.delay_sampler import ExponentialDelaySampler
from obp.simulator.simulator import BanditEnvironmentSimulator


def test_synthetic_sample_results_in_sampled_delay_when_delay_function_is_given():
    n_actions = 3
    delay_function = ExponentialDelaySampler(
        max_scale=100.0, random_state=12345
    ).exponential_delay_function

    dataset = BanditEnvironmentSimulator(
        n_actions=n_actions,
        reward_function=logistic_sparse_reward_function,
        delay_function=delay_function,
        random_state=12345,
    )

    actual_bandits_dataset = dataset.next_bandit_round_batch(n_rounds=5)

    expected_round_delays = np.tile([266.0, 39.0, 21.0, 23.0, 84.0], (n_actions, 1)).T
    assert (actual_bandits_dataset.round_delays == expected_round_delays).all()


def test_synthetic_sample_results_in_sampled_delay_with_weighted_delays_per_arm():
    n_actions = 3
    delay_function = ExponentialDelaySampler(
        max_scale=100.0, min_scale=10.0, random_state=12345
    ).exponential_delay_function_expected_reward_weighted

    dataset = BanditEnvironmentSimulator(
        n_actions=n_actions,
        reward_function=logistic_sparse_reward_function,
        delay_function=delay_function,
        random_state=12345,
    )

    actual_bandits_dataset = dataset.next_bandit_round_batch(n_rounds=1000)

    ordered_rewards = actual_bandits_dataset.expected_rewards[0].argsort()
    mean_delays = actual_bandits_dataset.round_delays.sum(axis=0)
    assert (
        mean_delays[ordered_rewards[2]]
        < mean_delays[ordered_rewards[1]]
        > mean_delays[ordered_rewards[2]]
    )


def test_synthetic_sample_results_with_exponential_delay_function_has_different_delays_each_batch():
    n_actions = 3

    delay_function = ExponentialDelaySampler(
        max_scale=1000.0, random_state=12345
    ).exponential_delay_function

    dataset = BanditEnvironmentSimulator(
        n_actions=n_actions,
        reward_function=logistic_sparse_reward_function,
        delay_function=delay_function,
        random_state=12345,
    )

    actual_delays_1 = dataset.next_bandit_round_batch(n_rounds=5).round_delays
    actual_delays_2 = dataset.next_bandit_round_batch(n_rounds=5).round_delays

    expected_round_delays_1 = np.tile(
        [2654.0, 381.0, 204.0, 229.0, 839.0], (n_actions, 1)
    ).T
    expected_round_delays_2 = np.tile(
        [906.0, 3339.0, 1059.0, 1382.0, 1061.0], (n_actions, 1)
    ).T

    assert (actual_delays_1 == expected_round_delays_1).all()
    assert (actual_delays_2 == expected_round_delays_2).all()


def test_synthetic_sample_results_with_exponential_delay_function_has_same_delays_each_dataset():
    n_actions = 3

    delay_function = ExponentialDelaySampler(
        max_scale=1000.0, random_state=12345
    ).exponential_delay_function

    dataset = BanditEnvironmentSimulator(
        n_actions=n_actions,
        reward_function=logistic_sparse_reward_function,
        delay_function=delay_function,
        random_state=12345,
    )

    actual_delays_1 = dataset.next_bandit_round_batch(n_rounds=5).round_delays

    delay_function = ExponentialDelaySampler(
        max_scale=1000.0, random_state=12345
    ).exponential_delay_function

    dataset = BanditEnvironmentSimulator(
        n_actions=n_actions,
        reward_function=logistic_sparse_reward_function,
        delay_function=delay_function,
        random_state=12345,
    )
    actual_delays_2 = dataset.next_bandit_round_batch(n_rounds=5).round_delays

    expected_round_delays_1 = np.tile(
        [2654.0, 381.0, 204.0, 229.0, 839.0], (n_actions, 1)
    ).T
    expected_round_delays_2 = np.tile(
        [2654.0, 381.0, 204.0, 229.0, 839.0], (n_actions, 1)
    ).T

    assert (actual_delays_1 == expected_round_delays_1).all()
    assert (actual_delays_2 == expected_round_delays_2).all()


def test_synthetic_sample_results_do_not_contain_reward_delay_when_delay_function_is_none():
    dataset = BanditEnvironmentSimulator(
        n_actions=3,
        reward_function=logistic_sparse_reward_function,
        random_state=12345,
    )

    actual_bandits_dataset = dataset.next_bandit_round_batch(n_rounds=5)

    assert actual_bandits_dataset.round_delays is None


def test_synthetic_sample_results_reward_delay_is_configurable_through_delay_function():
    def trivial_delay_func(*args, **kwargs):
        return np.asarray([1, 2, 3, 4, 5])

    dataset = BanditEnvironmentSimulator(
        n_actions=3,
        reward_function=logistic_sparse_reward_function,
        delay_function=trivial_delay_func,
        random_state=12345,
    )

    actual_bandits_dataset = dataset.next_bandit_round_batch(n_rounds=5)

    assert (actual_bandits_dataset.round_delays == [1, 2, 3, 4, 5]).all()


@pytest.mark.parametrize(
    "size, actions, random_state, expected_delays",
    [
        (5, 3, 12345, np.tile([266.0, 39.0, 21.0, 23.0, 84.0], (3, 1)).T),
        (3, 3, 12345, np.tile([266.0, 39.0, 21.0], (3, 1)).T),
        (5, 3, 54321, np.tile([243.0, 98.0, 157.0, 57.0, 79.0], (3, 1)).T),
    ],
)
def test_exponential_delay_function_results_in_expected_seeded_discrete_delays(
    size, actions, random_state, expected_delays
):
    delay_function = ExponentialDelaySampler(
        max_scale=100.0, random_state=random_state
    ).exponential_delay_function

    actual_delays = delay_function(size, n_actions=actions)
    assert (actual_delays == expected_delays).all()


@pytest.mark.parametrize(
    "size, actions, expected_reward, random_state, expected_delays",
    [
        (
            2,
            2,
            np.asarray([[1, 0.01], [0.5, 0.5]]),
            12344,
            np.asarray([[2.0, 55.0], [3.0, 27.0]]),
        ),
        (
            2,
            2,
            np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            12345,
            np.asarray([[242.0, 32.0], [15.0, 15.0]]),
        ),
    ],
)
def test_exponential_delay_function_conditioned_on_expected_reward_results_in_expected_seeded_discrete_delays(
    size, actions, expected_reward, random_state, expected_delays
):
    delay_function = ExponentialDelaySampler(
        max_scale=100.0, min_scale=10.0, random_state=random_state
    ).exponential_delay_function_expected_reward_weighted

    actual_delays = delay_function(expected_rewards=expected_reward)
    assert (actual_delays == expected_delays).all()
