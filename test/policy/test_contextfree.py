import pytest
import numpy as np

from obp.policy.contextfree import EpsilonGreedy
from obp.policy.contextfree import Random
from obp.policy.contextfree import BernoulliTS


def test_contextfree_base_exception():

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=0)

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions="3")

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=2, len_list=-1)

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=2, len_list="5")

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=2, batch_size=-3)

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=2, batch_size="3")


def test_egreedy_normal_epsilon():

    policy1 = EpsilonGreedy(n_actions=2)
    assert 0 <= policy1.epsilon <= 1

    policy2 = EpsilonGreedy(n_actions=3, epsilon=0.3)
    assert 0 <= policy2.epsilon <= 1


def test_egreedy_abnormal_epsilon():

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=2, epsilon=1.2)

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=5, epsilon=-0.2)


def test_egreedy_select_action_exploitation():
    trial_num = 50
    policy = EpsilonGreedy(n_actions=2, epsilon=0.0)
    policy.action_counts = np.array([3, 3])
    policy.reward_counts = np.array([3, 0])
    for _ in range(trial_num):
        assert policy.select_action()[0] == 0


def test_egreedy_select_action_exploration():
    trial_num = 50
    policy = EpsilonGreedy(n_actions=2, epsilon=1.0)
    policy.action_counts = np.array([3, 3])
    policy.reward_counts = np.array([3, 0])
    selected_action = [policy.select_action() for _ in range(trial_num)]
    assert 0 < sum(selected_action)[0] < trial_num


def test_egreedy_update_params():
    policy = EpsilonGreedy(n_actions=2, epsilon=1.0)
    policy.action_counts_temp = np.array([4, 3])
    policy.action_counts = np.copy(policy.action_counts_temp)
    policy.reward_counts_temp = np.array([2.0, 0.0])
    policy.reward_counts = np.copy(policy.reward_counts_temp)
    action = 0
    reward = 1.0
    policy.update_params(action, reward)
    assert np.array_equal(policy.action_counts, np.array([5, 3]))
    next_reward = (2.0 * (5 - 1) / 5) + (reward / 5)
    assert np.allclose(policy.reward_counts, np.array([next_reward, 0.0]))


def test_random_compute_batch_action_dist():
    n_actions = 10
    len_list = 5
    n_rounds = 100
    policy = Random(n_actions=n_actions, len_list=len_list)
    action_dist = policy.compute_batch_action_dist(n_rounds=n_rounds)
    assert action_dist.shape[0] == n_rounds
    assert action_dist.shape[1] == n_actions
    assert action_dist.shape[2] == len_list
    assert len(np.unique(action_dist)) == 1
    assert np.unique(action_dist)[0] == 1 / n_actions


def test_bernoulli_ts_zozotown_prior():
    with pytest.raises(Exception):
        BernoulliTS(n_actions=2, is_zozotown_prior=True)
