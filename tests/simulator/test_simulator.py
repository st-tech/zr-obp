import re

import pytest
from obp.policy.linear import LinTS

from obp.policy.contextfree import EpsilonGreedy
from obp.dataset.synthetic import logistic_reward_function
from obp.dataset import SyntheticBanditDataset
from obp.policy.policy_type import PolicyType
from obp.simulator import run_bandit_simulation


def test_run_bandit_simulation_updates_at_each_taken_action():
    n_rounds = 100

    dataset = SyntheticBanditDataset(
        n_actions=3,
        dim_context=5,
        reward_type="binary",
        reward_function=logistic_reward_function,
        random_state=12345,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    epsilon_greedy = EpsilonGreedy(n_actions=3)
    _ = run_bandit_simulation(bandit_feedback=bandit_feedback, policy=epsilon_greedy)

    assert epsilon_greedy.n_trial == n_rounds


def test_run_bandit_simulation_handles_context_in_simulations():
    n_rounds = 100

    dataset = SyntheticBanditDataset(
        n_actions=3,
        dim_context=5,
        reward_type="binary",
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

    dataset = SyntheticBanditDataset(
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
