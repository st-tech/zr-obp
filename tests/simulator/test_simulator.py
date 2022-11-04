import re

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from obp.policy import BaseContextFreePolicy, IPWLearner
from obp.policy.linear import LinTS, LinUCB

from obp.policy.contextfree import EpsilonGreedy, Random, BernoulliTS
from obp.dataset.synthetic import (
    logistic_reward_function,
    logistic_sparse_reward_function,
)
from obp.policy.policy_type import PolicyType
from obp.simulator.coefficient_drifter import CoefficientDrifter
from obp.simulator.delay_sampler import ExponentialDelaySampler
from obp.simulator.simulator import BanditEnvironmentSimulator, BanditPolicySimulator


def test_bandit_policy_simulator_updates_at_each_taken_action():
    n_rounds = 100

    env = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=5,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_rounds = env.next_bandit_round_batch(n_rounds=n_rounds)

    epsilon_greedy = EpsilonGreedy(n_actions=3)

    simulator = BanditPolicySimulator(
        policy=epsilon_greedy,
        environment=env,
    )

    simulator.steps(batch_bandit_rounds=bandit_rounds)

    assert epsilon_greedy.n_trial == n_rounds


def test_bandit_policy_simulator_handles_context_in_simulations():
    n_rounds = 100

    env = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=5,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_rounds = env.next_bandit_round_batch(n_rounds=n_rounds)

    lin_ts = LinTS(dim=env.dim_context, n_actions=env.n_actions, random_state=12345)
    simulator = BanditPolicySimulator(
        policy=lin_ts,
        environment=env,
    )

    simulator.steps(batch_bandit_rounds=bandit_rounds)

    assert lin_ts.n_trial == n_rounds


def test_bandit_policy_simulator_raises_on_unknown_policy():
    n_rounds = 1

    env = BanditEnvironmentSimulator(
        n_actions=3,
    )
    bandit_rounds = env.next_bandit_round_batch(n_rounds=n_rounds)

    class OfflineEpsilon(EpsilonGreedy):
        @property
        def policy_type(self) -> PolicyType:
            return PolicyType.OFFLINE

    epsilon_greedy = OfflineEpsilon(n_actions=3)
    simulator = BanditPolicySimulator(
        policy=epsilon_greedy,
        environment=env,
    )

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            r"Policy type PolicyType.OFFLINE of policy egreedy_1.0 is " r"unsupported"
        ),
    ):
        simulator.steps(batch_bandit_rounds=bandit_rounds)


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


def test_bandit_policy_simulator_works_end_to_end_with_synthetic_bandit_dataset():
    delay_function = ExponentialDelaySampler(
        max_scale=1.0, random_state=12345
    ).exponential_delay_function

    env = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=1,
        reward_function=logistic_reward_function,
        delay_function=delay_function,
        random_state=12345,
    )
    bandit_rounds = env.next_bandit_round_batch(n_rounds=5)

    policy = EpsilonGreedy(n_actions=3, epsilon=0.1, random_state=12345)

    simulator = BanditPolicySimulator(
        policy=policy,
        environment=env,
    )

    simulator.steps(batch_bandit_rounds=bandit_rounds)


def test_bandit_policy_simulator_applies_policy_in_delay_specified_order():
    n_rounds = 5

    env = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=1,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_rounds = env.next_bandit_round_batch(n_rounds=n_rounds)
    bandit_rounds.round_delays = np.tile([2, 1, 2, 1, 0], (3, 1)).T

    tracker = BanditUpdateTracker(n_actions=3, random_state=12345)

    simulator = BanditPolicySimulator(
        policy=tracker,
        environment=env,
    )

    simulator.steps(batch_bandit_rounds=bandit_rounds)

    expected_updates = [
        {"round": 3, "action": 0, "reward": 1},
        {"round": 3, "action": 0, "reward": 1},
        {"round": 5, "action": 2, "reward": 1},
        {"round": 5, "action": 1, "reward": 0},
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
def test_bandit_policy_simulator_does_not_crash_with_various_bandit_algorithms(policy):
    n_rounds = 5

    env = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=4,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_rounds = env.next_bandit_round_batch(n_rounds=n_rounds)
    bandit_rounds.round_delays = np.tile([2, 1, 2, 2, 2], (3, 1)).T

    simulator = BanditPolicySimulator(
        policy=policy,
        environment=env,
    )

    simulator.steps(batch_bandit_rounds=bandit_rounds)


def test_bandit_policy_simulator_applies_policy_directly_when_no_delay():
    n_rounds = 5

    env = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=1,
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_rounds = env.next_bandit_round_batch(n_rounds=n_rounds)
    bandit_rounds.round_delays = None

    tracker = BanditUpdateTracker(n_actions=3, random_state=12345)

    simulator = BanditPolicySimulator(
        policy=tracker,
        environment=env,
    )
    simulator.steps(batch_bandit_rounds=bandit_rounds)

    expected_updates = [
        {"action": 0, "reward": 1, "round": 1},
        {"action": 0, "reward": 1, "round": 2},
        {"action": 2, "reward": 1, "round": 3},
        {"action": 1, "reward": 0, "round": 4},
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
        base_coefficient_weight=0.3,
        random_state=1234,
    )

    env_1 = BanditEnvironmentSimulator(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_function=logistic_sparse_reward_function,
        delay_function=None,
        coef_function=drifter.get_coefficients,
        random_state=12345,
    )

    policy = EpsilonGreedy(n_actions=n_actions, epsilon=0.1, random_state=12345)
    simulator_1 = BanditPolicySimulator(
        policy=policy,
        environment=env_1,
    )
    simulator_1.steps(n_rounds=sample_n_rounds)

    drifter = CoefficientDrifter(
        drift_interval=drift_interval,
        transition_period=transition_period,
        transition_type="linear",  # linear or weighted_sampled
        seasonal=False,
        base_coefficient_weight=0.3,
        random_state=1234,
    )

    env_2 = BanditEnvironmentSimulator(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_function=logistic_sparse_reward_function,
        delay_function=None,
        coef_function=drifter.get_coefficients,
        random_state=12345,
    )

    policy = EpsilonGreedy(n_actions=n_actions, epsilon=0.1, random_state=12345)
    simulator_2 = BanditPolicySimulator(
        policy=policy,
        environment=env_2,
    )
    simulator_2.steps(n_rounds=sample_n_rounds)

    assert simulator_1.total_reward == simulator_2.total_reward


def test_bandit_environment_simulator_fetches_new_context_every_pull():
    env = BanditEnvironmentSimulator(
        n_actions=10,
        dim_context=6,
        reward_function=logistic_sparse_reward_function,
        delay_function=None,
        random_state=12345,
    )

    contexts = [env.next_context() for _ in range(5)]
    assert len(np.hstack(contexts)[0]) == len(np.unique(contexts))


def test_bandit_environment_simulator_can_fetch_a_batch_of_contexts():
    env = BanditEnvironmentSimulator(
        n_actions=10,
        dim_context=6,
        reward_function=logistic_sparse_reward_function,
        delay_function=None,
        random_state=12345,
    )

    contexts = env.next_context_batch(n_rounds=5)
    assert contexts.shape == (5, 6)


def test_bandit_environment_simulator_can_a_single_step():
    env = BanditEnvironmentSimulator(
        n_actions=10,
        dim_context=6,
        reward_function=logistic_sparse_reward_function,
        delay_function=None,
        random_state=12345,
    )

    _ = env.next_bandit_round()


def test_bandit_policy_simulator_can_a_single_steps_and_keep_track():
    env = BanditEnvironmentSimulator(
        n_actions=10,
        dim_context=6,
        reward_function=logistic_sparse_reward_function,
        delay_function=None,
        random_state=12345,
    )

    simulator = BanditPolicySimulator(
        policy=EpsilonGreedy(epsilon=0.1, random_state=12345, n_actions=10),
        environment=env,
    )

    for i in range(100):
        assert simulator.rounds_played == i
        assert len(simulator.selected_actions) == i
        simulator.step()

    assert simulator.total_reward > 1


def test_bandit_policy_simulator_can_do_multiple_steps_in_call_and_keep_track_of_actions_and_performance():
    env = BanditEnvironmentSimulator(
        n_actions=10,
        dim_context=6,
        reward_function=logistic_sparse_reward_function,
        delay_function=None,
        random_state=12345,
    )

    simulator = BanditPolicySimulator(
        policy=EpsilonGreedy(epsilon=0.1, random_state=12345, n_actions=10),
        environment=env,
    )

    simulator.steps(n_rounds=100)

    assert simulator.total_reward > 1
    assert simulator.rounds_played == 100
    assert len(simulator.selected_actions) == 100
    assert len(simulator.obtained_rewards) == 100


def test_bandit_policy_simulator_can_update_policy_with_delays_if_delay_rounds_are_available():
    class MockBanditEnvironmentSimulator(BanditEnvironmentSimulator):
        round_delays = np.tile([1, 1, 2, 1, 0], (5, 1)).T

        def next_bandit_round(self):
            bandit_round = super().next_bandit_round()
            bandit_round.round_delays, self.round_delays = (
                self.round_delays[-1],
                self.round_delays[:-1],
            )
            return bandit_round

    env = MockBanditEnvironmentSimulator(
        n_actions=3,
        dim_context=1,
        reward_function=logistic_reward_function,
        random_state=12345,
    )

    tracker = BanditUpdateTracker(n_actions=3, random_state=12345)

    simulator = BanditPolicySimulator(
        policy=tracker,
        environment=env,
    )

    simulator.steps(n_rounds=5)

    expected_updates = [1, 3, 5, 5]

    assert [update["round"] for update in tracker.parameter_updates] == expected_updates


def test_bandit_policy_simulator_clears_delay_queue_when_called_into_last_available_round():
    class MockBanditEnvironmentSimulator(BanditEnvironmentSimulator):
        round_delays = np.tile([1, 1, 4, 3, 2], (5, 1)).T

        def next_bandit_round(self):
            bandit_round = super().next_bandit_round()
            bandit_round.round_delays, self.round_delays = (
                self.round_delays[-1],
                self.round_delays[:-1],
            )
            return bandit_round

    env = MockBanditEnvironmentSimulator(
        n_actions=3,
        dim_context=1,
        reward_function=logistic_reward_function,
        random_state=12345,
    )

    tracker = BanditUpdateTracker(n_actions=3, random_state=12345)

    simulator = BanditPolicySimulator(
        policy=tracker,
        environment=env,
    )

    simulator.steps(n_rounds=5)

    expected_updates_before_queue_cleared = [3, 5, 5]
    assert [
        update["round"] for update in tracker.parameter_updates
    ] == expected_updates_before_queue_cleared

    simulator.clear_delayed_queue()

    expected_updates_before_queue_cleared = [3, 5, 5, 5, 5]

    assert len(simulator.reward_round_lookup.values()) == 0
    assert [
        update["round"] for update in tracker.parameter_updates
    ] == expected_updates_before_queue_cleared


def test_bandit_policy_simulator_do_simulation_over_batch_data():
    env = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=1,
        reward_function=logistic_reward_function,
        random_state=12345,
    )

    simulator = BanditPolicySimulator(
        policy=EpsilonGreedy(n_actions=3, epsilon=0.1, random_state=12345),
        environment=env,
    )

    simulator.steps(batch_bandit_rounds=env.next_bandit_round_batch(5))

    assert simulator.rounds_played == 5


def test_bandit_policy_simulator_cleans_up_when_simulation_is_interupted():
    env = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=4,
        reward_function=logistic_reward_function,
        random_state=12345,
    )

    class CrashingBanditPolicySimulator(BanditPolicySimulator):
        def _step(self):
            super()._step()
            if self.rounds_played == 50:
                raise RuntimeError

    simulator = CrashingBanditPolicySimulator(
        policy=EpsilonGreedy(n_actions=3, epsilon=0.1, random_state=12345),
        environment=env,
    )

    with pytest.raises(RuntimeError):
        simulator.steps(batch_bandit_rounds=env.next_bandit_round_batch(100))

    assert simulator.rounds_played == 50
    assert simulator.contexts.shape == (50, 4)
    assert simulator.ground_truth_rewards.shape == (50, 3)


def test_bandit_policy_simulator_cleans_up_keeping_previous_rounds_when_simulation_is_interupted():
    env = BanditEnvironmentSimulator(
        n_actions=3,
        dim_context=4,
        reward_function=logistic_reward_function,
        random_state=12345,
    )

    class CrashingBanditPolicySimulator(BanditPolicySimulator):
        def _step(self):
            super()._step()
            if self.rounds_played == 5:
                raise RuntimeError

    simulator = CrashingBanditPolicySimulator(
        policy=EpsilonGreedy(n_actions=3, epsilon=0.1, random_state=12345),
        environment=env,
    )

    batch_1 = env.next_bandit_round_batch(2)
    simulator.steps(batch_bandit_rounds=batch_1)

    batch_2 = env.next_bandit_round_batch(6)
    with pytest.raises(RuntimeError):
        simulator.steps(batch_bandit_rounds=batch_2)

    assert simulator.rounds_played == 5
    assert np.allclose(
        simulator.contexts,
        np.vstack((batch_1.context, batch_2.context))[
            :5,
        ],
    )
    assert np.allclose(
        simulator.ground_truth_rewards,
        np.vstack((batch_1.rewards, batch_2.rewards))[
            :5,
        ],
    )


def test_ipw_can_be_learned_from_logged_data_generated_by_simulation():
    from sklearn.ensemble import RandomForestClassifier as RandomForest

    env = BanditEnvironmentSimulator(
        n_actions=10,
        dim_context=5,
        reward_function=logistic_reward_function,
        random_state=12345,
    )

    simulator = BanditPolicySimulator(
        policy=EpsilonGreedy(n_actions=10, epsilon=0.1, random_state=12345),
        environment=env,
    )

    simulator.steps(batch_bandit_rounds=env.next_bandit_round_batch(100))

    propensity_model = LogisticRegression(random_state=12345)
    propensity_model.fit(simulator.contexts, simulator.selected_actions)
    pscores = propensity_model.predict_proba(simulator.contexts)

    ipw_learner = IPWLearner(
        n_actions=env.n_actions,
        base_classifier=RandomForest(
            n_estimators=30, min_samples_leaf=10, random_state=12345
        ),
    )

    ipw_learner.fit(
        context=simulator.contexts,
        action=simulator.selected_actions,
        reward=simulator.obtained_rewards,
        pscore=np.choose(simulator.selected_actions, pscores.T),
    )
    eval_action_dists = ipw_learner.predict(context=simulator.contexts)

    rewards = np.sum(
        simulator.ground_truth_rewards * np.squeeze(eval_action_dists, axis=-1)
    )
    assert rewards > simulator.total_reward
