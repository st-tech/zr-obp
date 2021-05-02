import pytest
import numpy as np

from obp.policy.linear import LinEpsilonGreedy
from obp.policy.linear import LinUCB
from obp.policy.linear import LinTS
from obp.policy.policy_type import PolicyType


def test_linear_base_exception():
    # invalid dim
    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim=-3)

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim=0)

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim="3")

    # invalid n_actions
    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=-3, dim=2)

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=1, dim=2)

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions="2", dim=2)

    # invalid len_list
    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim=2, len_list=-3)

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim=2, len_list=0)

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim=2, len_list="3")

    # invalid batch_size
    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim=2, batch_size=-2)

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim=2, batch_size=0)

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim=2, batch_size="10")

    # invalid relationship between n_actions and len_list
    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=5, len_list=10, dim=2)

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, len_list=3, dim=2)


def test_lin_epsilon_normal_epsilon():

    policy1 = LinEpsilonGreedy(n_actions=2, dim=2)
    assert 0 <= policy1.epsilon <= 1

    policy2 = LinEpsilonGreedy(n_actions=2, dim=2, epsilon=0.3)
    assert policy2.epsilon == 0.3


def test_lin_epsilon_abnormal_epsilon():

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim=2, epsilon=1.2)

    with pytest.raises(ValueError):
        LinEpsilonGreedy(n_actions=2, dim=2, epsilon=-0.2)


def test_lin_epsilon_select_action_exploitation():
    trial_num = 50
    policy = LinEpsilonGreedy(n_actions=2, dim=2, epsilon=0.0)

    # policy type
    assert policy.policy_type == PolicyType.CONTEXTUAL

    context = np.array([1.0, 1.0]).reshape(1, -1)
    policy.update_params(action=0, reward=1.0, context=context)
    policy.update_params(action=0, reward=1.0, context=context)
    policy.update_params(action=1, reward=1.0, context=context)
    policy.update_params(action=1, reward=0.0, context=context)
    for _ in range(trial_num):
        assert policy.select_action(context=context)[0] == 0


def test_lin_epsilon_select_action_exploration():
    trial_num = 50
    policy = LinEpsilonGreedy(n_actions=2, dim=2, epsilon=1.0)
    context = np.array([1.0, 1.0]).reshape(1, -1)
    policy.update_params(action=0, reward=1.0, context=context)
    policy.update_params(action=0, reward=1.0, context=context)
    policy.update_params(action=1, reward=1.0, context=context)
    policy.update_params(action=1, reward=0.0, context=context)
    selected_action = [policy.select_action(context=context) for _ in range(trial_num)]
    assert 0 < sum(selected_action)[0] < trial_num


def test_lin_epsilon_update_params():
    # check the consistency with Sherman–Morrison formula
    policy = LinEpsilonGreedy(n_actions=2, dim=2, epsilon=1.0)
    action = 0
    reward = 1.0
    context = np.array([1, 0]).reshape(1, -1)
    A_inv_temp = np.array([[1 / 2, 0], [0, 1]])
    b_temp = np.array([1, 1])
    policy.A_inv_temp[action] = np.copy(A_inv_temp)
    policy.b_temp[:, action] = np.copy(b_temp)
    policy.update_params(action=action, reward=reward, context=context)
    next_A_inv = A_inv_temp - np.array([[1 / 4, 0], [0, 0]]) / (1 + 1 / 2)
    next_b = b_temp + reward * context
    assert np.allclose(policy.A_inv[action], next_A_inv)
    assert np.allclose(policy.b[:, action], next_b)


def test_lin_ucb_initialize():
    # note that the meaning of epsilon is different from that of LinEpsilonGreedy
    with pytest.raises(ValueError):
        LinUCB(n_actions=2, dim=2, epsilon=-0.2)

    n_actions = 3
    dim = 2
    policy = LinUCB(n_actions=n_actions, dim=dim, epsilon=2.0)
    assert policy.theta_hat.shape == (dim, n_actions)
    assert policy.A_inv.shape == (n_actions, dim, dim)
    assert policy.b.shape == (dim, n_actions)
    assert policy.A_inv_temp.shape == (n_actions, dim, dim)
    assert policy.b_temp.shape == (dim, n_actions)


def test_lin_ucb_select_action():
    dim = 3
    len_list = 2
    policy = LinUCB(n_actions=4, dim=dim, len_list=2, epsilon=0.0)
    context = np.ones(dim).reshape(1, -1)
    action = policy.select_action(context=context)
    assert len(action) == len_list


def test_lin_ucb_update_params():
    # check the consistency with Sherman–Morrison formula
    policy = LinUCB(n_actions=2, dim=2, epsilon=1.0)
    action = 0
    reward = 1.0
    context = np.array([1, 0]).reshape(1, -1)
    A_inv_temp = np.array([[1 / 2, 0], [0, 1]])
    b_temp = np.array([1, 1])
    policy.A_inv_temp[action] = np.copy(A_inv_temp)
    policy.b_temp[:, action] = np.copy(b_temp)
    policy.update_params(action=action, reward=reward, context=context)
    next_A_inv = A_inv_temp - np.array([[1 / 4, 0], [0, 0]]) / (1 + 1 / 2)
    next_b = b_temp + reward * context
    assert np.allclose(policy.A_inv[action], next_A_inv)
    assert np.allclose(policy.b[:, action], next_b)


def test_lin_ts_initialize():
    n_actions = 3
    dim = 2
    policy = LinTS(n_actions=n_actions, dim=dim)
    assert policy.A_inv.shape == (n_actions, dim, dim)
    assert policy.b.shape == (dim, n_actions)
    assert policy.A_inv_temp.shape == (n_actions, dim, dim)
    assert policy.b_temp.shape == (dim, n_actions)


def test_lin_ts_select_action():
    dim = 3
    len_list = 2
    policy = LinTS(n_actions=4, dim=dim, len_list=2)
    context = np.ones(dim).reshape(1, -1)
    action = policy.select_action(context=context)
    assert len(action) == len_list


def test_lin_ts_update_params():
    # check the consistency with Sherman–Morrison formula
    policy = LinTS(n_actions=2, dim=2)
    action = 0
    reward = 1.0
    context = np.array([1, 0]).reshape(1, -1)
    A_inv_temp = np.array([[1 / 2, 0], [0, 1]])
    b_temp = np.array([1, 1])
    policy.A_inv_temp[action] = np.copy(A_inv_temp)
    policy.b_temp[:, action] = np.copy(b_temp)
    policy.update_params(action=action, reward=reward, context=context)
    next_A_inv = A_inv_temp - np.array([[1 / 4, 0], [0, 0]]) / (1 + 1 / 2)
    next_b = b_temp + reward * context
    assert np.allclose(policy.A_inv[action], next_A_inv)
    assert np.allclose(policy.b[:, action], next_b)
