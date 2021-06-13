import pytest
import numpy as np

from obp.policy.contextfree import EpsilonGreedy
from obp.policy.contextfree import Random
from obp.policy.contextfree import BernoulliTS
from obp.policy.policy_type import PolicyType


def test_contextfree_base_exception():
    # invalid n_actions
    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=0)

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions="3")

    # invalid len_list
    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=2, len_list=-1)

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=2, len_list="5")

    # invalid batch_size
    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=2, batch_size=-3)

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=2, batch_size="3")

    # invalid relationship between n_actions and len_list
    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=5, len_list=10)

    with pytest.raises(ValueError):
        EpsilonGreedy(n_actions=2, len_list=3)


def test_egreedy_normal_epsilon():

    policy1 = EpsilonGreedy(n_actions=2)
    assert 0 <= policy1.epsilon <= 1

    policy2 = EpsilonGreedy(n_actions=3, epsilon=0.3)
    assert 0 <= policy2.epsilon <= 1

    # policy type
    assert EpsilonGreedy(n_actions=2).policy_type == PolicyType.CONTEXT_FREE


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
    assert np.allclose(policy.reward_counts, np.array([2.0 + reward, 0.0]))


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

    policy_all = BernoulliTS(n_actions=2, is_zozotown_prior=True, campaign="all")
    # check whether it is not an non-informative prior parameter (i.e., default parameter)
    assert len(np.unique(policy_all.alpha)) != 1
    assert len(np.unique(policy_all.beta)) != 1

    policy_men = BernoulliTS(n_actions=2, is_zozotown_prior=True, campaign="men")
    assert len(np.unique(policy_men.alpha)) != 1
    assert len(np.unique(policy_men.beta)) != 1

    policy_women = BernoulliTS(n_actions=2, is_zozotown_prior=True, campaign="women")
    assert len(np.unique(policy_women.alpha)) != 1
    assert len(np.unique(policy_women.beta)) != 1


def test_bernoulli_ts_select_action():
    # invalid relationship between n_actions and len_list
    with pytest.raises(ValueError):
        BernoulliTS(n_actions=5, len_list=10)

    with pytest.raises(ValueError):
        BernoulliTS(n_actions=2, len_list=3)

    policy1 = BernoulliTS(n_actions=3, len_list=3)
    assert np.allclose(np.sort(policy1.select_action()), np.array([0, 1, 2]))

    policy = BernoulliTS(n_actions=5, len_list=3)
    assert len(policy.select_action()) == 3


def test_bernoulli_ts_update_params():
    policy = BernoulliTS(n_actions=2)
    policy.action_counts_temp = np.array([4, 3])
    policy.action_counts = np.copy(policy.action_counts_temp)
    policy.reward_counts_temp = np.array([2.0, 0.0])
    policy.reward_counts = np.copy(policy.reward_counts_temp)
    action = 0
    reward = 1.0
    policy.update_params(action, reward)
    assert np.array_equal(policy.action_counts, np.array([5, 3]))
    # in bernoulli ts, reward_counts is defined as the sum of observed rewards for each action
    next_reward = 2.0 + reward
    assert np.allclose(policy.reward_counts, np.array([next_reward, 0.0]))


def test_bernoulli_ts_compute_batch_action_dist():
    n_rounds = 10
    n_actions = 5
    len_list = 2
    policy = BernoulliTS(n_actions=n_actions, len_list=len_list)
    action_dist = policy.compute_batch_action_dist(n_rounds=n_rounds, n_sim=30)
    assert action_dist.shape[0] == n_rounds
    assert action_dist.shape[1] == n_actions
    assert action_dist.shape[2] == len_list
