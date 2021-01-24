import pytest
import numpy as np

from obp.policy.logistic import LogisticEpsilonGreedy
from obp.policy.logistic import LogisticUCB
from obp.policy.logistic import LogisticTS
from obp.policy.logistic import MiniBatchLogisticRegression


def test_logistic_epsilon_normal_epsilon():

    policy1 = LogisticEpsilonGreedy(n_actions=2, dim=2)
    assert 0 <= policy1.epsilon <= 1

    policy2 = LogisticEpsilonGreedy(n_actions=2, dim=2, epsilon=0.5)
    assert policy2.epsilon == 0.5


def test_logistic_epsilon_abnormal_epsilon():

    with pytest.raises(ValueError):
        LogisticEpsilonGreedy(n_actions=2, dim=2, epsilon=1.3)

    with pytest.raises(ValueError):
        LogisticEpsilonGreedy(n_actions=2, dim=2, epsilon=-0.3)


def test_logistic_epsilon_each_action_model():
    n_actions = 3
    policy = LogisticEpsilonGreedy(n_actions=n_actions, dim=2, epsilon=0.5)
    for i in range(n_actions):
        assert isinstance(policy.model_list[i], MiniBatchLogisticRegression)


def test_logistic_epsilon_select_action_exploitation():
    trial_num = 50
    policy = LogisticEpsilonGreedy(n_actions=2, dim=2, epsilon=0.0)
    context = np.array([1.0, 1.0]).reshape(1, -1)
    policy.update_params(action=0, reward=1.0, context=context)
    policy.update_params(action=0, reward=1.0, context=context)
    policy.update_params(action=1, reward=1.0, context=context)
    policy.update_params(action=1, reward=0.0, context=context)
    for _ in range(trial_num):
        assert policy.select_action(context=context)[0] == 0


def test_logistic_epsilon_select_action_exploration():
    trial_num = 50
    policy = LogisticEpsilonGreedy(n_actions=2, dim=2, epsilon=1.0)
    context = np.array([1.0, 1.0]).reshape(1, -1)
    policy.update_params(action=0, reward=1.0, context=context)
    policy.update_params(action=0, reward=1.0, context=context)
    policy.update_params(action=1, reward=1.0, context=context)
    policy.update_params(action=1, reward=0.0, context=context)
    selected_action = [policy.select_action(context=context) for _ in range(trial_num)]
    assert 0 < sum(selected_action)[0] < trial_num


def test_logistic_ucb_initialize():
    # note that the meaning of epsilon is different from that of LogisticEpsilonGreedy
    with pytest.raises(ValueError):
        LogisticUCB(n_actions=2, dim=2, epsilon=-0.2)

    n_actions = 3
    policy = LogisticUCB(n_actions=n_actions, dim=2, epsilon=0.5)
    for i in range(n_actions):
        assert isinstance(policy.model_list[i], MiniBatchLogisticRegression)


def test_logistic_ucb_select_action():
    dim = 3
    len_list = 2
    policy = LogisticUCB(n_actions=4, dim=dim, len_list=2, epsilon=0.0)
    context = np.ones(dim).reshape(1, -1)
    action = policy.select_action(context=context)
    assert len(action) == len_list


def test_logistic_ts_initialize():
    n_actions = 3
    policy = LogisticTS(n_actions=n_actions, dim=2)
    for i in range(n_actions):
        assert isinstance(policy.model_list[i], MiniBatchLogisticRegression)


def test_logistic_ts_select_action():
    dim = 3
    len_list = 2
    policy = LogisticTS(n_actions=4, dim=dim, len_list=2)
    context = np.ones(dim).reshape(1, -1)
    action = policy.select_action(context=context)
    assert len(action) == len_list
