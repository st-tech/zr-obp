import pytest
import numpy as np

from obp.dataset import SyntheticContinuousBanditDataset
from obp.dataset.synthetic_continuous import (
    linear_reward_funcion_continuous,
    quadratic_reward_funcion_continuous,
    linear_behavior_policy_continuous,
    linear_synthetic_policy_continuous,
    threshold_synthetic_policy_continuous,
    sign_synthetic_policy_continuous,
)


# dim_context, action_noise, reward_noise, min_action_value, max_action_value, random_state, err, description
invalid_input_of_init = [
    (
        0,  #
        1.0,
        1.0,
        -1.0,
        1.0,
        12345,
        ValueError,
        "`dim_context`= 0, must be >= 1.",
    ),
    (
        1.0,  #
        1.0,
        1.0,
        -1.0,
        1.0,
        12345,
        TypeError,
        "`dim_context` must be an instance of <class 'int'>, not <class 'float'>.",
    ),
    (
        "3",  #
        1.0,
        1.0,
        -1.0,
        1.0,
        12345,
        TypeError,
        "`dim_context` must be an instance of <class 'int'>, not <class 'str'>.",
    ),
    (
        None,  #
        1.0,
        1.0,
        -1.0,
        1.0,
        12345,
        TypeError,
        "`dim_context` must be an instance of <class 'int'>, not <class 'NoneType'>.",
    ),
    (
        3,
        -1.0,  #
        1.0,
        -1.0,
        1.0,
        12345,
        ValueError,
        "`action_noise`= -1.0, must be >= 0.",
    ),
    (
        3,
        "3",  #
        1.0,
        -1.0,
        1.0,
        12345,
        TypeError,
        r"`action_noise` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        3,
        None,  #
        1.0,
        -1.0,
        1.0,
        12345,
        TypeError,
        r"`action_noise` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (
        3,
        1.0,
        -1.0,  #
        -1.0,
        1.0,
        12345,
        ValueError,
        "`reward_noise`= -1.0, must be >= 0.",
    ),
    (
        3,
        1.0,
        "3",  #
        -1.0,
        1.0,
        12345,
        TypeError,
        r"`reward_noise` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        3,
        1.0,
        None,  #
        -1.0,
        1.0,
        12345,
        TypeError,
        r"`reward_noise` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (
        3,
        1.0,
        1.0,
        "3",  #
        1.0,
        12345,
        TypeError,
        r"`min_action_value` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        3,
        1.0,
        1.0,
        None,  #
        1.0,
        12345,
        TypeError,
        r"`min_action_value` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (
        3,
        1.0,
        1.0,
        1.0,
        "3",  #
        12345,
        TypeError,
        r"`max_action_value` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        3,
        1.0,
        1.0,
        1.0,
        None,  #
        12345,
        TypeError,
        r"`max_action_value` must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (
        3,
        1.0,
        1.0,
        1.0,  #
        -1.0,  #
        12345,
        ValueError,
        "`max_action_value` must be larger than `min_action_value`",
    ),
    (
        3,
        1.0,
        1.0,
        -1.0,
        1.0,
        None,
        ValueError,
        "random_state must be given",
    ),
    (
        3,
        1.0,
        1.0,
        -1.0,
        1.0,
        "",
        ValueError,
        "'' cannot be used to seed a numpy.random.RandomState instance",
    ),
]


@pytest.mark.parametrize(
    "dim_context, action_noise, reward_noise, min_action_value, max_action_value, random_state, err, description",
    invalid_input_of_init,
)
def test_synthetic_continuous_init_using_invalid_inputs(
    dim_context,
    action_noise,
    reward_noise,
    min_action_value,
    max_action_value,
    random_state,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = SyntheticContinuousBanditDataset(
            dim_context=dim_context,
            action_noise=action_noise,
            reward_noise=reward_noise,
            min_action_value=min_action_value,
            max_action_value=max_action_value,
            random_state=random_state,
        )


# n_rounds, err, description
invalid_input_of_obtain_batch_bandit_feedback = [
    (
        0,  #
        ValueError,
        "`n_rounds`= 0, must be >= 1.",
    ),
    (
        1.0,  #
        TypeError,
        "`n_rounds` must be an instance of <class 'int'>, not <class 'float'>.",
    ),
    (
        "3",  #
        TypeError,
        "`n_rounds` must be an instance of <class 'int'>, not <class 'str'>.",
    ),
    (
        None,  #
        TypeError,
        "`n_rounds` must be an instance of <class 'int'>, not <class 'NoneType'>.",
    ),
]


@pytest.mark.parametrize(
    "n_rounds, err, description",
    invalid_input_of_obtain_batch_bandit_feedback,
)
def test_synthetic_continuous_obtain_batch_bandit_feedback_using_invalid_inputs(
    n_rounds,
    err,
    description,
):
    dataset = SyntheticContinuousBanditDataset()

    with pytest.raises(err, match=f"{description}*"):
        _ = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)


def test_synthetic_continuous_obtain_batch_bandit_feedback():
    # bandit feedback
    n_rounds = 10
    min_action_value = -1.0
    max_action_value = 1.0
    dataset = SyntheticContinuousBanditDataset(
        min_action_value=min_action_value,
        max_action_value=max_action_value,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds,
    )
    assert bandit_feedback["n_rounds"] == n_rounds
    assert (
        bandit_feedback["context"].shape[0] == n_rounds  # n_rounds
        and bandit_feedback["context"].shape[1] == 1  # default dim_context
    )
    assert (
        bandit_feedback["action"].ndim == 1
        and len(bandit_feedback["action"]) == n_rounds
    )
    assert np.all(min_action_value <= bandit_feedback["action"]) and np.all(
        bandit_feedback["action"] <= max_action_value
    )
    assert bandit_feedback["position"] is None
    assert (
        bandit_feedback["reward"].ndim == 1
        and len(bandit_feedback["reward"]) == n_rounds
    )
    assert (
        bandit_feedback["expected_reward"].ndim == 1
        and len(bandit_feedback["expected_reward"]) == n_rounds
    )
    assert (
        bandit_feedback["pscore"].ndim == 1
        and len(bandit_feedback["pscore"]) == n_rounds
    )


# context, action, description
invalid_input_of_calc_policy_value = [
    (
        np.ones((3, 1)),
        np.ones(4),
        "the size of axis 0 of context must be the same as that of action",
    ),
    (
        np.ones((4, 2)),  #
        np.ones(4),
        "the size of axis 1 of context must be the same as dim_context",
    ),
    ("3", np.ones(4), "context must be 2-dimensional ndarray"),
    (None, np.ones(4), "context must be 2-dimensional ndarray"),
    (
        np.ones((4, 1)),
        np.ones((4, 1)),  #
        "action must be 1-dimensional ndarray",
    ),
    (np.ones((4, 1)), "3", "action must be 1-dimensional ndarray"),
    (np.ones((4, 1)), None, "action must be 1-dimensional ndarray"),
]


@pytest.mark.parametrize(
    "context, action, description",
    invalid_input_of_calc_policy_value,
)
def test_synthetic_continuous_calc_policy_value_using_invalid_inputs(
    context,
    action,
    description,
):
    dataset = SyntheticContinuousBanditDataset()

    with pytest.raises(ValueError, match=f"{description}*"):
        _ = dataset.calc_ground_truth_policy_value(
            context=context,
            action=action,
        )


def test_synthetic_continuous_calc_policy_value():
    n_rounds = 10
    dim_context = 3
    dataset = SyntheticContinuousBanditDataset(
        dim_context=dim_context,
        min_action_value=1,
        max_action_value=10,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds,
    )

    policy_value = dataset.calc_ground_truth_policy_value(
        context=bandit_feedback["context"],
        action=bandit_feedback["action"],
    )
    assert isinstance(
        policy_value, float
    ), "Invalid response of calc_ground_truth_policy_value"
    assert policy_value == bandit_feedback["expected_reward"].mean()


def test_synthetic_linear_reward_funcion_continuous():
    # context
    with pytest.raises(ValueError):
        context = np.array([1.0, 1.0])
        linear_reward_funcion_continuous(context=context, action=np.ones(2))

    with pytest.raises(ValueError):
        context = [1.0, 1.0]
        linear_reward_funcion_continuous(context=context, action=np.ones([2, 2]))

    # action
    with pytest.raises(ValueError):
        action = np.array([1.0])
        linear_reward_funcion_continuous(context=np.ones([2, 2]), action=action)

    with pytest.raises(ValueError):
        action = [1.0, 1.0]
        linear_reward_funcion_continuous(context=np.ones([2, 2]), action=action)

    with pytest.raises(ValueError):
        linear_reward_funcion_continuous(context=np.ones([2, 2]), action=np.ones(3))

    # expected_reward
    n_rounds = 10
    dim_context = 3
    context = np.ones([n_rounds, dim_context])
    action = np.ones(n_rounds)
    expected_reward = linear_reward_funcion_continuous(context=context, action=action)
    assert expected_reward.shape[0] == n_rounds and expected_reward.ndim == 1


def test_synthetic_quadratic_reward_funcion_continuous():
    # context
    with pytest.raises(ValueError):
        context = np.array([1.0, 1.0])
        quadratic_reward_funcion_continuous(context=context, action=np.ones(2))

    with pytest.raises(ValueError):
        context = [1.0, 1.0]
        quadratic_reward_funcion_continuous(context=context, action=np.ones([2, 2]))

    # action
    with pytest.raises(ValueError):
        action = np.array([1.0])
        quadratic_reward_funcion_continuous(context=np.ones([2, 2]), action=action)

    with pytest.raises(ValueError):
        action = [1.0, 1.0]
        quadratic_reward_funcion_continuous(context=np.ones([2, 2]), action=action)

    with pytest.raises(ValueError):
        quadratic_reward_funcion_continuous(context=np.ones([2, 2]), action=np.ones(3))

    # expected_reward
    n_rounds = 10
    dim_context = 3
    context = np.ones([n_rounds, dim_context])
    action = np.ones(n_rounds)
    expected_reward = quadratic_reward_funcion_continuous(
        context=context, action=action
    )
    assert expected_reward.shape[0] == n_rounds and expected_reward.ndim == 1


def test_synthetic_linear_behavior_policy_continuous():
    # context
    with pytest.raises(ValueError):
        context = np.array([1.0, 1.0])
        linear_behavior_policy_continuous(context=context)

    with pytest.raises(ValueError):
        context = [1.0, 1.0]
        linear_behavior_policy_continuous(context=context)

    # expected continuous action values by behavior policy
    n_rounds = 10
    dim_context = 3
    context = np.ones([n_rounds, dim_context])
    expected_continuous_actions = linear_behavior_policy_continuous(context=context)
    assert (
        expected_continuous_actions.shape[0] == n_rounds
        and expected_continuous_actions.ndim == 1
    )


def test_linear_synthetic_policy_continuous():
    # context
    with pytest.raises(ValueError):
        context = np.array([1.0, 1.0])
        linear_behavior_policy_continuous(context=context)

    with pytest.raises(ValueError):
        context = [1.0, 1.0]
        linear_behavior_policy_continuous(context=context)

    # continuous action values given by a synthetic policy
    n_rounds = 10
    dim_context = 3
    context = np.ones([n_rounds, dim_context])
    continuous_actions = linear_synthetic_policy_continuous(context=context)
    assert continuous_actions.shape[0] == n_rounds and continuous_actions.ndim == 1


def test_threshold_synthetic_policy_continuous():
    # context
    with pytest.raises(ValueError):
        context = np.array([1.0, 1.0])
        threshold_synthetic_policy_continuous(context=context)

    with pytest.raises(ValueError):
        context = [1.0, 1.0]
        threshold_synthetic_policy_continuous(context=context)

    # continuous action values given by a synthetic policy
    n_rounds = 10
    dim_context = 3
    context = np.ones([n_rounds, dim_context])
    continuous_actions = threshold_synthetic_policy_continuous(context=context)
    assert continuous_actions.shape[0] == n_rounds and continuous_actions.ndim == 1


def test_sign_synthetic_policy_continuous():
    # context
    with pytest.raises(ValueError):
        context = np.array([1.0, 1.0])
        sign_synthetic_policy_continuous(context=context)

    with pytest.raises(ValueError):
        context = [1.0, 1.0]
        sign_synthetic_policy_continuous(context=context)

    # continuous action values given by a synthetic policy
    n_rounds = 10
    dim_context = 3
    context = np.ones([n_rounds, dim_context])
    continuous_actions = sign_synthetic_policy_continuous(context=context)
    assert continuous_actions.shape[0] == n_rounds and continuous_actions.ndim == 1
