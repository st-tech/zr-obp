import re

import numpy as np
import pytest

from obp.policy import BaseContextFreePolicy
from obp.policy.linear import LinTS, LinUCB

from obp.policy.contextfree import EpsilonGreedy, Random, BernoulliTS
from obp.dataset.synthetic import logistic_reward_function, logistic_sparse_reward_function
from obp.policy.policy_type import PolicyType
from obp.simulator import run_bandit_simulation
from obp.simulator.coefficient_drifter import CoefficientDrifter
from obp.simulator.delay_sampler import ExponentialDelaySampler
from obp.simulator.simulator import BanditEnvironmentSimulator, count_ground_truth_policy_rewards


def test_run_bandit_simulation_updates_at_each_taken_action():
    n_rounds = 100

    dataset = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=5,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    epsilon_greedy = EpsilonGreedy(n_actions=3)
    _ = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=epsilon_greedy)

    assert epsilon_greedy.n_trial == n_rounds


def test_run_bandit_simulation_handles_context_in_simulations():
    n_rounds = 100

    dataset = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=5,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    lin_ts = LinTS(
        dim=dataset.dim_context, n_actions=dataset.n_actions, random_state=12345
    )
    _ = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=lin_ts)

    assert lin_ts.n_trial == n_rounds


def test_run_bandit_simulation_raises_on_unknown_policy():
    n_rounds = 1

    dataset = BanditEnvironmentSimulator(
        n_actions=3,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    class OfflineEpsilon(EpsilonGreedy):
        @property
        def policy_type(self) -> PolicyType:
            return PolicyType.OFFLINE

    epsilon_greedy = OfflineEpsilon(n_actions=3)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            r"Policy type PolicyType.OFFLINE of policy egreedy_1.0 is " r"unsupported"
        ),
    ):
        _ = run_bandit_simulation(
            bandit_feedback=bandit_feedback, policy=epsilon_greedy
        )


class BanditUpdateTracker(BaseContextFreePolicy):
    """
    This class can be used to keep track on updates being sent to the policy.
    The policy itself always provides a random choice and should not be used for any analytical purposes.
    """

    n_rounds = 0

    def __post_init__(self) -> None:
        self.parameter_updates = []
        super().__post_init__()

    def select_action(self) -> np.ndarray:
        self.n_rounds += 1
        return self.random_.choice(self.n_actions, size=self.len_list, replace=False)

    def update_params(self, action: int, reward: float) -> None:
        self.parameter_updates.append(
            {"round": self.n_rounds, "action": action, "reward": reward}
        )


def test_run_bandit_simulation_works_end_to_end_with_synthetic_bandit_dataset():
    delay_function = ExponentialDelaySampler(
        max_scale=1.0, random_state=12345
    ).exponential_delay_function

    dataset = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=1,
        reward_function=logistic_reward_function,
        delay_function=delay_function,
        random_state=12345,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=5)

    policy = EpsilonGreedy(n_actions=3, epsilon=0.1, random_state=12345)
    _ = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=policy)


def test_run_bandit_simulation_applies_policy_in_delay_specified_order():
    n_rounds = 5

    dataset = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=1,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    bandit_feedback["round_delays"] = np.tile([2, 1, 2, 1, 0], (3, 1)).T

    tracker = BanditUpdateTracker(n_actions=3, random_state=12345)
    _ = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=tracker)

    expected_updates = [
        {"round": 3, "action": 0, "reward": 0},
        {"round": 3, "action": 0, "reward": 1},
        {"round": 5, "action": 2, "reward": 1},
        {"round": 5, "action": 1, "reward": 1},
        {"round": 5, "action": 2, "reward": 0},
    ]

    assert tracker.parameter_updates == expected_updates


def test_run_bandit_simulation_applies_all_rewards_delayed_till_after_all_rounds_to_the_end_of_simulation():
    n_rounds = 5

    dataset = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=1,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    bandit_feedback["round_delays"] = np.tile([2, 1, 2, 2, 2], (3, 1)).T

    tracker = BanditUpdateTracker(n_actions=3, random_state=12345)
    _ = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=tracker)

    expected_updates = [
        {"round": 3, "action": 0, "reward": 0},
        {"round": 3, "action": 0, "reward": 1},
        {"round": 5, "action": 2, "reward": 1},
        {"round": 5, "action": 1, "reward": 1},
        {"round": 5, "action": 2, "reward": 0},
    ]

    assert tracker.parameter_updates == expected_updates


@pytest.mark.parametrize(
    "policy",
    [
        Random(n_actions=3, epsilon=1.0, random_state=12345),
        EpsilonGreedy(n_actions=3, epsilon=0.1, random_state=12345),
        BernoulliTS(n_actions=3, random_state=12345),
        LinTS(dim=4, n_actions=3, random_state=12345),
        LinUCB(dim=4, n_actions=3, random_state=12345),
    ],
)
def test_run_bandit_simulation_does_not_crash_with_various_bandit_algorithms(policy):
    n_rounds = 5

    dataset = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=4,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    bandit_feedback["round_delays"] = np.tile([2, 1, 2, 2, 2], (3, 1)).T

    _ = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=policy)


def test_run_bandit_simulation_applies_policy_directly_when_no_delay():
    n_rounds = 5

    dataset = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=1,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    bandit_feedback["round_delays"] = None

    tracker = BanditUpdateTracker(n_actions=3, random_state=12345)
    _ = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=tracker)

    expected_updates = [
        {"action": 0, "reward": 0, "round": 1},
        {"action": 0, "reward": 1, "round": 2},
        {"action": 2, "reward": 1, "round": 3},
        {"action": 1, "reward": 1, "round": 4},
        {"action": 2, "reward": 0, "round": 5},
    ]

    assert tracker.parameter_updates == expected_updates


def test_simulator_can_create_identical_simulations_using_seeds():
    sample_n_rounds = 40
    drift_interval = 20
    transition_period = 0
    n_actions = 3
    dim_context = 5

    drifter = CoefficientDrifter(
        drift_interval=drift_interval,
        transition_period=transition_period,
        transition_type="linear",  # linear or weighted_sampled
        seasonal=False,
        base_coefficient_weight=.3,
        random_state=1234
    )

    dataset_1 = BanditEnvironmentSimulator(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_function=logistic_sparse_reward_function,
        delay_function=None,
        coef_function=drifter.get_coefficients,
        random_state=12345,
    )

    training_bandit_dataset_1 = dataset_1.obtain_batch_bandit_feedback(n_rounds=sample_n_rounds)

    policy = EpsilonGreedy(n_actions=n_actions, epsilon=0.1, random_state=12345)
    train_action_dists_1 = run_bandit_simulation(
        bandit_feedback=training_bandit_dataset_1,
        policy=policy
    )

    rewards_1 = count_ground_truth_policy_rewards(train_action_dists_1.squeeze(axis=2)[:drift_interval],
                                                            training_bandit_dataset_1["rewards"][
                                                            :drift_interval])
    drifter = CoefficientDrifter(
        drift_interval=drift_interval,
        transition_period=transition_period,
        transition_type="linear",  # linear or weighted_sampled
        seasonal=False,
        base_coefficient_weight=1.0,
        random_state=1234
    )


    dataset_2 = BanditEnvironmentSimulator(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_function=logistic_sparse_reward_function,
        delay_function=None,
        coef_function=drifter.get_coefficients,
        random_state=12345,
    )

    training_bandit_dataset_2 = dataset_2.obtain_batch_bandit_feedback(n_rounds=sample_n_rounds)

    policy = EpsilonGreedy(n_actions=n_actions, epsilon=0.1, random_state=12345)
    train_action_dists_2 = run_bandit_simulation(
        bandit_feedback=training_bandit_dataset_2,
        policy=policy
    )

    rewards_2 = count_ground_truth_policy_rewards(train_action_dists_2.squeeze(axis=2)[:drift_interval],
                                                      training_bandit_dataset_2["rewards"][:drift_interval])

    assert rewards_1 == rewards_2
