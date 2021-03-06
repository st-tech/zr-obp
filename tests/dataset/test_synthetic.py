import pytest
import numpy as np

from obp.dataset import SyntheticBanditDataset
from obp.dataset.synthetic import (
    logistic_reward_function,
    linear_reward_function,
    linear_behavior_policy,
)


def test_synthetic_init():
    # n_actions
    with pytest.raises(ValueError):
        SyntheticBanditDataset(n_actions=1)

    with pytest.raises(ValueError):
        SyntheticBanditDataset(n_actions="3")

    # dim_context
    with pytest.raises(ValueError):
        SyntheticBanditDataset(n_actions=2, dim_context=0)

    with pytest.raises(ValueError):
        SyntheticBanditDataset(n_actions=2, dim_context="2")

    # reward_type
    with pytest.raises(ValueError):
        SyntheticBanditDataset(n_actions=2, reward_type="aaa")

    # when reward_function is None, expected_reward is randomly sampled in [0, 1]
    # this check includes the test of `sample_contextfree_expected_reward` function
    dataset = SyntheticBanditDataset(n_actions=2)
    assert len(dataset.expected_reward) == 2
    assert np.all(0 <= dataset.expected_reward) and np.all(dataset.expected_reward <= 1)

    # when behavior_policy_function is None, behavior_policy is set to uniform one
    uniform_policy = np.array([0.5, 0.5])
    assert np.allclose(dataset.behavior_policy, uniform_policy)

    # action_context
    ohe = np.eye(2, dtype=int)
    assert np.allclose(dataset.action_context, ohe)


def test_synthetic_obtain_batch_bandit_feedback():
    # n_rounds
    with pytest.raises(ValueError):
        dataset = SyntheticBanditDataset(n_actions=2)
        dataset.obtain_batch_bandit_feedback(n_rounds=0)

    with pytest.raises(ValueError):
        dataset = SyntheticBanditDataset(n_actions=2)
        dataset.obtain_batch_bandit_feedback(n_rounds="3")

    # bandit feedback
    n_rounds = 10
    n_actions = 5
    dataset = SyntheticBanditDataset(n_actions=n_actions)
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    assert bandit_feedback["n_rounds"] == n_rounds
    assert bandit_feedback["n_actions"] == n_actions
    assert (
        bandit_feedback["context"].shape[0] == n_rounds  # n_rounds
        and bandit_feedback["context"].shape[1] == 1  # default dim_context
    )
    assert (
        bandit_feedback["action_context"].shape[0] == n_actions
        and bandit_feedback["action_context"].shape[1] == n_actions
    )
    assert (
        bandit_feedback["action"].ndim == 1
        and len(bandit_feedback["action"]) == n_rounds
    )
    assert (
        bandit_feedback["position"].ndim == 1
        and len(bandit_feedback["position"]) == n_rounds
    )
    assert (
        bandit_feedback["reward"].ndim == 1
        and len(bandit_feedback["reward"]) == n_rounds
    )
    assert (
        bandit_feedback["expected_reward"].shape[0] == n_rounds
        and bandit_feedback["expected_reward"].shape[1] == n_actions
    )
    assert (
        bandit_feedback["pscore"].ndim == 1
        and len(bandit_feedback["pscore"]) == n_rounds
    )


def test_synthetic_logistic_reward_function():
    # context
    with pytest.raises(ValueError):
        context = np.array([1.0, 1.0])
        logistic_reward_function(context=context, action_context=np.ones([2, 2]))

    with pytest.raises(ValueError):
        context = [1.0, 1.0]
        logistic_reward_function(context=context, action_context=np.ones([2, 2]))

    # action_context
    with pytest.raises(ValueError):
        action_context = np.array([1.0, 1.0])
        logistic_reward_function(context=np.ones([2, 2]), action_context=action_context)

    with pytest.raises(ValueError):
        action_context = [1.0, 1.0]
        logistic_reward_function(context=np.ones([2, 2]), action_context=action_context)

    # expected_reward
    n_rounds = 10
    dim_context = dim_action_context = 3
    n_actions = 5
    context = np.ones([n_rounds, dim_context])
    action_context = np.ones([n_actions, dim_action_context])
    expected_reward = logistic_reward_function(
        context=context, action_context=action_context
    )
    assert (
        expected_reward.shape[0] == n_rounds and expected_reward.shape[1] == n_actions
    )
    assert np.all(0 <= expected_reward) and np.all(expected_reward <= 1)


def test_synthetic_linear_reward_function():
    # context
    with pytest.raises(ValueError):
        context = np.array([1.0, 1.0])
        linear_reward_function(context=context, action_context=np.ones([2, 2]))

    with pytest.raises(ValueError):
        context = [1.0, 1.0]
        linear_reward_function(context=context, action_context=np.ones([2, 2]))

    # action_context
    with pytest.raises(ValueError):
        action_context = np.array([1.0, 1.0])
        linear_reward_function(context=np.ones([2, 2]), action_context=action_context)

    with pytest.raises(ValueError):
        action_context = [1.0, 1.0]
        linear_reward_function(context=np.ones([2, 2]), action_context=action_context)

    # expected_reward
    n_rounds = 10
    dim_context = dim_action_context = 3
    n_actions = 5
    context = np.ones([n_rounds, dim_context])
    action_context = np.ones([n_actions, dim_action_context])
    expected_reward = linear_reward_function(
        context=context, action_context=action_context
    )
    assert (
        expected_reward.shape[0] == n_rounds and expected_reward.shape[1] == n_actions
    )


def test_synthetic_linear_behavior_policy():
    # context
    with pytest.raises(ValueError):
        context = np.array([1.0, 1.0])
        linear_behavior_policy(context=context, action_context=np.ones([2, 2]))

    with pytest.raises(ValueError):
        context = [1.0, 1.0]
        linear_behavior_policy(context=context, action_context=np.ones([2, 2]))

    # action_context
    with pytest.raises(ValueError):
        action_context = np.array([1.0, 1.0])
        linear_behavior_policy(context=np.ones([2, 2]), action_context=action_context)

    with pytest.raises(ValueError):
        action_context = [1.0, 1.0]
        linear_behavior_policy(context=np.ones([2, 2]), action_context=action_context)

    # expected_reward
    n_rounds = 10
    dim_context = dim_action_context = 3
    n_actions = 5
    context = np.ones([n_rounds, dim_context])
    action_context = np.ones([n_actions, dim_action_context])
    action_prob = linear_behavior_policy(context=context, action_context=action_context)
    assert action_prob.shape[0] == n_rounds and action_prob.shape[1] == n_actions
    assert np.all(0 <= action_prob) and np.all(action_prob <= 1)
