import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import torch

from obp.policy.offline import IPWLearner
from obp.policy.offline import NNPolicyLearner
from obp.policy.offline import QLearner
from obp.policy.policy_type import PolicyType


base_classifier = LogisticRegression()
base_regressor = LinearRegression()

# n_actions, len_list, base_classifier, description
invalid_input_of_ipw_learner_init = [
    (
        0,  #
        1,
        base_classifier,
        "n_actions == 0, must be >= 1",
    ),
    (
        10,
        -1,  #
        base_classifier,
        "len_list == -1, must be >= 0",
    ),
    (
        10,
        20,  #
        base_classifier,
        "len_list == 20, must be <= 10",
    ),
    (10, 1, base_regressor, "`base_classifier` must be a classifier"),
]

valid_input_of_ipw_learner_init = [
    (
        10,
        1,
        None,
        "valid input",
    ),
    (
        10,
        1,
        base_classifier,
        "valid input",
    ),
]


@pytest.mark.parametrize(
    "n_actions, len_list, base_classifier, description",
    invalid_input_of_ipw_learner_init,
)
def test_ipw_learner_init_using_invalid_inputs(
    n_actions,
    len_list,
    base_classifier,
    description,
):
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = IPWLearner(
            n_actions=n_actions,
            len_list=len_list,
            base_classifier=base_classifier,
        )


@pytest.mark.parametrize(
    "n_actions, len_list, base_classifier, description",
    valid_input_of_ipw_learner_init,
)
def test_ipw_learner_init_using_valid_inputs(
    n_actions,
    len_list,
    base_classifier,
    description,
):
    ipw_learner = IPWLearner(
        n_actions=n_actions,
        len_list=len_list,
        base_classifier=base_classifier,
    )
    # policy_type
    assert ipw_learner.policy_type == PolicyType.OFFLINE


def test_ipw_learner_init_base_classifier_list():
    # base classifier
    len_list = 2
    learner1 = IPWLearner(n_actions=2, len_list=len_list)
    assert isinstance(learner1.base_classifier, LogisticRegression)
    for i in range(len_list):
        assert isinstance(learner1.base_classifier_list[i], LogisticRegression)

    from sklearn.naive_bayes import GaussianNB

    learner2 = IPWLearner(n_actions=2, len_list=len_list, base_classifier=GaussianNB())
    assert isinstance(learner2.base_classifier, GaussianNB)
    for i in range(len_list):
        assert isinstance(learner2.base_classifier_list[i], GaussianNB)


def test_ipw_learner_create_train_data_for_opl():
    context = np.array([1.0, 1.0]).reshape(1, -1)
    learner = IPWLearner(n_actions=2)
    action = np.array([0])
    reward = np.array([1.0])
    pscore = np.array([0.5])

    X, sample_weight, y = learner._create_train_data_for_opl(
        context=context, action=action, reward=reward, pscore=pscore
    )

    assert np.allclose(X, np.array([1.0, 1.0]).reshape(1, -1))
    assert np.allclose(sample_weight, np.array([2.0]))
    assert np.allclose(y, np.array([0]))


def test_ipw_learner_fit():
    n_rounds = 1000
    dim_context = 5
    n_actions = 3
    len_list = 2
    context = np.ones((n_rounds, dim_context))
    action = np.random.choice(np.arange(len_list, dtype=int), size=n_rounds)
    reward = np.random.choice(np.arange(2), size=n_rounds)
    position = np.random.choice(np.arange(len_list, dtype=int), size=n_rounds)

    # inconsistency with the shape
    desc = "Expected `context.shape[0]"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = IPWLearner(n_actions=n_actions, len_list=len_list)
        variant_context = np.random.normal(size=(n_rounds + 1, n_actions))
        learner.fit(
            context=variant_context,
            action=action,
            reward=reward,
            position=position,
        )

    # len_list > 2, but position is not set
    desc = "When `self.len_list > 1"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = IPWLearner(n_actions=n_actions, len_list=len_list)
        learner.fit(context=context, action=action, reward=reward)

    # position must be non-negative
    desc = "`position` elements must be non-negative integers"
    with pytest.raises(ValueError, match=f"{desc}*"):
        negative_position = position - 1
        learner = IPWLearner(n_actions=n_actions, len_list=len_list)
        learner.fit(
            context=context, action=action, reward=reward, position=negative_position
        )

    # IPWLearner cannot handle negative rewards
    desc = "A negative value is found in"
    with pytest.raises(ValueError, match=f"{desc}*"):
        negative_reward = reward - 1.0
        learner = IPWLearner(n_actions=n_actions, len_list=len_list)
        learner.fit(
            context=context, action=action, reward=negative_reward, position=position
        )


def test_ipw_learner_predict():
    n_actions = 2
    len_list = 1

    # shape error
    desc = "`context` must be 2D array"
    with pytest.raises(ValueError, match=f"{desc}*"):
        context = np.array([1.0, 1.0])
        learner = IPWLearner(n_actions=n_actions, len_list=len_list)
        learner.predict(context=context)

    # shape consistency of action_dist
    # n_rounds is 5, dim_context is 2
    context = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, -1)
    action = np.array([0, 1])
    reward = np.array([1.0, 0.0])
    position = np.array([0, 0])
    learner = IPWLearner(n_actions=2, len_list=1)
    learner.fit(context=context, action=action, reward=reward, position=position)

    context_test = np.array([i for i in range(10)]).reshape(5, 2)
    action_dist = learner.predict(context=context_test)
    assert np.allclose(
        action_dist.sum(1), np.ones_like((context_test.shape[0], len_list))
    )
    assert action_dist.shape[0] == 5
    assert action_dist.shape[1] == n_actions
    assert action_dist.shape[2] == len_list


def test_ipw_learner_sample_action():
    n_actions = 2
    len_list = 1
    context = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, -1)
    action = np.array([0, 1])
    reward = np.array([1.0, 0.0])
    position = np.array([0, 0])
    learner = IPWLearner(n_actions=n_actions, len_list=len_list)
    learner.fit(context=context, action=action, reward=reward, position=position)

    desc = "`context` must be 2D array"
    with pytest.raises(ValueError, match=f"{desc}*"):
        invalid_type_context = [1.0, 2.0]
        learner.sample_action(context=invalid_type_context)

    with pytest.raises(ValueError, match=f"{desc}*"):
        invalid_ndim_context = np.array([1.0, 2.0, 3.0, 4.0])
        learner.sample_action(context=invalid_ndim_context)

    context = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, -1)
    n_rounds = context.shape[0]
    sampled_action = learner.sample_action(context=context)

    assert sampled_action.shape[0] == n_rounds
    assert sampled_action.shape[1] == n_actions
    assert sampled_action.shape[2] == len_list


# n_actions, len_list, base_model, fitting_method, description
invalid_input_of_q_learner_init = [
    (
        0,  #
        1,
        base_classifier,
        "normal",
        "n_actions == 0, must be >= 1",
    ),
    (
        10,
        -1,  #
        base_classifier,
        "normal",
        "len_list == -1, must be >= 0",
    ),
    (
        10,
        20,  #
        base_classifier,
        "normal",
        "len_list == 20, must be <= 10",
    ),
    (10, 1, "base_regressor", "normal", "`base_model` must be BaseEstimator"),  #
    (
        10,
        1,
        base_classifier,
        "None",  #
        "`fitting_method` must be one of 'normal', 'iw', or 'mrdr', but",
    ),
]

valid_input_of_q_learner_init = [
    (
        10,
        1,
        base_classifier,
        "normal",
        "valid input",
    ),
    (
        10,
        1,
        base_classifier,
        "iw",
        "valid input",
    ),
    (
        10,
        1,
        base_regressor,
        "normal",
        "valid input",
    ),
]


@pytest.mark.parametrize(
    "n_actions, len_list, base_model, fitting_method, description",
    invalid_input_of_q_learner_init,
)
def test_q_learner_init_using_invalid_inputs(
    n_actions,
    len_list,
    base_model,
    fitting_method,
    description,
):
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = QLearner(
            n_actions=n_actions,
            len_list=len_list,
            base_model=base_model,
            fitting_method=fitting_method,
        )


@pytest.mark.parametrize(
    "n_actions, len_list, base_model, fitting_method, description",
    valid_input_of_q_learner_init,
)
def test_q_learner_init_using_valid_inputs(
    n_actions,
    len_list,
    base_model,
    fitting_method,
    description,
):
    q_learner = QLearner(
        n_actions=n_actions,
        len_list=len_list,
        base_model=base_model,
        fitting_method=fitting_method,
    )
    # policy_type
    assert q_learner.policy_type == PolicyType.OFFLINE


def test_q_learner_fit():
    n_rounds = 1000
    dim_context = 5
    n_actions = 3
    len_list = 2
    context = np.ones((n_rounds, dim_context))
    action = np.random.choice(np.arange(len_list, dtype=int), size=n_rounds)
    reward = np.random.choice(np.arange(2), size=n_rounds)
    position = np.random.choice(np.arange(len_list, dtype=int), size=n_rounds)

    # inconsistency with the shape
    desc = "Expected `context.shape[0]"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = QLearner(
            n_actions=n_actions, len_list=len_list, base_model=base_classifier
        )
        variant_context = np.random.normal(size=(n_rounds + 1, n_actions))
        learner.fit(
            context=variant_context,
            action=action,
            reward=reward,
            position=position,
        )

    # len_list > 2, but position is not set
    desc = "When `self.len_list > 1"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = QLearner(
            n_actions=n_actions, len_list=len_list, base_model=base_classifier
        )
        learner.fit(context=context, action=action, reward=reward)

    # position must be non-negative
    desc = "`position` elements must be non-negative integers"
    with pytest.raises(ValueError, match=f"{desc}*"):
        negative_position = position - 1
        learner = QLearner(
            n_actions=n_actions, len_list=len_list, base_model=base_classifier
        )
        learner.fit(
            context=context, action=action, reward=reward, position=negative_position
        )


def test_q_learner_predict():
    n_actions = 2
    len_list = 1

    # shape error
    desc = "`context` must be 2D array"
    with pytest.raises(ValueError, match=f"{desc}*"):
        context = np.array([1.0, 1.0])
        learner = QLearner(
            n_actions=n_actions, len_list=len_list, base_model=base_classifier
        )
        learner.predict(context=context)

    # shape consistency of action_dist
    # n_rounds is 5, dim_context is 2
    context = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, -1)
    action = np.array([0, 1])
    reward = np.array([1.0, 0.0])
    position = np.array([0, 0])
    learner = QLearner(
        n_actions=n_actions, len_list=len_list, base_model=base_classifier
    )
    learner.fit(context=context, action=action, reward=reward, position=position)

    context_test = np.array([i for i in range(10)]).reshape(5, 2)
    action_dist = learner.predict(context=context_test)
    assert np.allclose(
        action_dist.sum(1), np.ones_like((context_test.shape[0], len_list))
    )
    assert action_dist.shape[0] == 5
    assert action_dist.shape[1] == n_actions
    assert action_dist.shape[2] == len_list


def test_q_learner_sample_action():
    n_actions = 2
    len_list = 1
    context = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, -1)
    action = np.array([0, 1])
    reward = np.array([1.0, 0.0])
    position = np.array([0, 0])
    learner = QLearner(
        n_actions=n_actions, len_list=len_list, base_model=base_classifier
    )
    learner.fit(context=context, action=action, reward=reward, position=position)

    desc = "`context` must be 2D array"
    with pytest.raises(ValueError, match=f"{desc}*"):
        invalid_type_context = [1.0, 2.0]
        learner.sample_action(context=invalid_type_context)

    with pytest.raises(ValueError, match=f"{desc}*"):
        invalid_ndim_context = np.array([1.0, 2.0, 3.0, 4.0])
        learner.sample_action(context=invalid_ndim_context)

    context = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, -1)
    n_rounds = context.shape[0]
    sampled_action = learner.sample_action(context=context)

    assert sampled_action.shape[0] == n_rounds
    assert sampled_action.shape[1] == n_actions
    assert sampled_action.shape[2] == len_list


# n_actions, len_list, dim_context, off_policy_objective, policy_reg_param, var_reg_param, hidden_layer_size,
# activation, solver, alpha, batch_size, learning_rate_init, max_iter, shuffle, random_state, tol,
# momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change,
# err, description
invalid_input_of_nn_policy_learner_init = [
    (
        0,  #
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "n_actions == 0, must be >= 1",
    ),
    (
        10,
        -1,  #
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "len_list == -1, must be >= 0",
    ),
    (
        10,
        1,
        -1,  #
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "dim_context == -1, must be >= 1.",
    ),
    (
        10,
        1,
        2,
        None,  #
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`off_policy_objective` must be one of 'dm', 'ipw', 'dr'",
    ),
    (
        10,
        1,
        2,
        "dros",  #
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`off_policy_objective` must be one of 'dm', 'ipw', 'dr'",
    ),
    (
        10,
        1,
        2,
        "snipw",
        -1.0,  #
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "lambda_ == -1.0, must be >= 0.",
    ),
    (
        10,
        1,
        2,
        "ipw-os",
        -1.0,  #
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "lambda_ == -1.0, must be >= 0.",
    ),
    (
        10,
        1,
        2,
        "ipw-subgauss",
        -1.0,  #
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "lambda_ == -1.0, must be >= 0.",
    ),
    (
        10,
        1,
        2,
        "ipw-subgauss",
        2.0,  #
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "lambda_ == 2.0, must be <= 1.",
    ),
    (
        10,
        1,
        2,
        "ipw-subgauss",
        "-1.0",  #
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        TypeError,
        r"lambda_ must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        10,
        1,
        2,
        "dr",
        0.0,
        "",  #
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        TypeError,
        r"policy_reg_param must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        10,
        1,
        2,
        "dr",
        0.0,
        None,  #
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        TypeError,
        r"policy_reg_param must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (
        10,
        1,
        2,
        "dr",
        0.0,
        -1.0,  #
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        r"policy_reg_param == -1.0, must be >= 0.0.",
    ),
    (
        10,
        1,
        2,
        "dr",
        0.0,
        0.1,
        "",  #
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        TypeError,
        r"var_reg_param must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        10,
        1,
        2,
        "dr",
        0.0,
        0.1,
        None,  #
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        TypeError,
        r"var_reg_param must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'NoneType'>.",
    ),
    (
        10,
        1,
        2,
        "dr",
        0.0,
        0.1,
        -1.0,  #
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        r"var_reg_param == -1.0, must be >= 0.0.",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, ""),  #
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`hidden_layer_size` must be a tuple of positive integers",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "None",  #
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`activation` must be one of 'identity', 'logistic', 'tanh', 'relu', or 'elu'",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "None",  #
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`solver` must be one of 'adam', 'adagrad', or 'sgd'",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        -1.0,  #
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "alpha == -1.0, must be >= 0.0",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        0,  #
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`batch_size` must be a positive integer or 'auto'",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0,  #
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`learning_rate_init`= 0.0, must be > 0.0",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        0,  #
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "max_iter == 0, must be >= 1",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        None,  #
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`shuffle` must be a bool",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        "",  #
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "'' cannot be used to seed",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        -1.0,  #
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`tol`= -1.0, must be > 0.0",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        2.0,  #
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "momentum == 2.0, must be <= 1.0",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        "",  #
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`nesterovs_momentum` must be a bool",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        None,  #
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`early_stopping` must be a bool",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "lbfgs",  #
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,  #
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "`solver` must be one of 'adam', 'adagrad', or 'sgd',",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        2.0,  #
        0.9,
        0.999,
        1e-8,
        10,
        ValueError,
        "validation_fraction == 2.0, must be <= 1.0",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        2.0,  #
        0.999,
        1e-8,
        10,
        ValueError,
        "beta_1 == 2.0, must be <= 1.0",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        2.0,  #
        1e-8,
        10,
        ValueError,
        "beta_2 == 2.0, must be <= 1.0",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        -1.0,  #
        10,
        ValueError,
        "epsilon == -1.0, must be >= 0.0",
    ),
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        0,  #
        ValueError,
        "n_iter_no_change == 0, must be >= 1.",
    ),
]

valid_input_of_nn_policy_learner_init = [
    (
        10,
        1,
        2,
        "ipw",
        0.0,
        0.1,
        0.1,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        "valid input",
    ),
    (
        10,
        1,
        2,
        "dr",
        0.0,
        0.1,
        0.1,
        (100,),
        "logistic",
        "sgd",
        0.001,
        50,
        0.0001,
        200,
        True,
        None,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        "valid input",
    ),
    (
        10,
        1,
        2,
        "snipw",
        1.0,
        0.1,
        0.1,
        (100,),
        "logistic",
        "sgd",
        0.001,
        50,
        0.0001,
        200,
        True,
        None,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        "valid input",
    ),
    (
        10,
        1,
        2,
        "ipw-os",
        100,
        0.1,
        0.1,
        (100,),
        "logistic",
        "sgd",
        0.001,
        50,
        0.0001,
        200,
        True,
        None,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        "valid input",
    ),
    (
        10,
        1,
        2,
        "ipw-subgauss",
        0.5,
        0.1,
        0.1,
        (100,),
        "logistic",
        "sgd",
        0.001,
        50,
        0.0001,
        200,
        True,
        None,
        1e-4,
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        "valid input",
    ),
]


@pytest.mark.parametrize(
    "n_actions, len_list, dim_context, off_policy_objective, lambda_, policy_reg_param, var_reg_param, hidden_layer_size, activation, solver, alpha, batch_size, learning_rate_init, max_iter, shuffle, random_state, tol, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, err, description",
    invalid_input_of_nn_policy_learner_init,
)
def test_nn_policy_learner_init_using_invalid_inputs(
    n_actions,
    len_list,
    dim_context,
    off_policy_objective,
    lambda_,
    policy_reg_param,
    var_reg_param,
    hidden_layer_size,
    activation,
    solver,
    alpha,
    batch_size,
    learning_rate_init,
    max_iter,
    shuffle,
    random_state,
    tol,
    momentum,
    nesterovs_momentum,
    early_stopping,
    validation_fraction,
    beta_1,
    beta_2,
    epsilon,
    n_iter_no_change,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=dim_context,
            off_policy_objective=off_policy_objective,
            lambda_=lambda_,
            policy_reg_param=policy_reg_param,
            var_reg_param=var_reg_param,
            hidden_layer_size=hidden_layer_size,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
        )


@pytest.mark.parametrize(
    "n_actions, len_list, dim_context, off_policy_objective, lambda_, policy_reg_param, var_reg_param, hidden_layer_size, activation, solver, alpha, batch_size, learning_rate_init, max_iter, shuffle, random_state, tol, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, description",
    valid_input_of_nn_policy_learner_init,
)
def test_nn_policy_learner_init_using_valid_inputs(
    n_actions,
    len_list,
    dim_context,
    off_policy_objective,
    lambda_,
    policy_reg_param,
    var_reg_param,
    hidden_layer_size,
    activation,
    solver,
    alpha,
    batch_size,
    learning_rate_init,
    max_iter,
    shuffle,
    random_state,
    tol,
    momentum,
    nesterovs_momentum,
    early_stopping,
    validation_fraction,
    beta_1,
    beta_2,
    epsilon,
    n_iter_no_change,
    description,
):
    nn_policy_learner = NNPolicyLearner(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=dim_context,
        off_policy_objective=off_policy_objective,
        lambda_=lambda_,
        policy_reg_param=policy_reg_param,
        var_reg_param=var_reg_param,
        hidden_layer_size=hidden_layer_size,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        shuffle=shuffle,
        random_state=random_state,
        tol=tol,
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        n_iter_no_change=n_iter_no_change,
    )
    assert isinstance(nn_policy_learner, NNPolicyLearner)


def test_nn_policy_learner_create_train_data_for_opl():
    context = np.ones((100, 2), dtype=np.int32)
    action = np.zeros((100,), dtype=np.int32)
    reward = np.ones((100,), dtype=np.float32)
    pscore = np.array([0.5] * 100, dtype=np.float32)
    estimated_rewards_by_reg_model = np.ones((100, 2), dtype=np.float32)
    position = np.zeros((100,), dtype=np.int32)

    learner1 = NNPolicyLearner(n_actions=2, dim_context=2, off_policy_objective="ipw")

    training_loader, validation_loader = learner1._create_train_data_for_opl(
        context=context,
        action=action,
        reward=reward,
        pscore=pscore,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        position=position,
    )

    assert isinstance(training_loader, torch.utils.data.DataLoader)
    assert validation_loader is None

    learner2 = NNPolicyLearner(
        n_actions=2,
        dim_context=2,
        off_policy_objective="ipw",
        early_stopping=True,
    )

    training_loader, validation_loader = learner2._create_train_data_for_opl(
        context=context,
        action=action,
        reward=reward,
        pscore=pscore,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        position=position,
    )

    assert isinstance(training_loader, torch.utils.data.DataLoader)
    assert isinstance(validation_loader, torch.utils.data.DataLoader)


def test_nn_policy_learner_fit():
    context = np.ones((100, 2), dtype=np.float32)
    action = np.zeros((100,), dtype=int)
    reward = np.ones((100,), dtype=np.float32)
    pscore = np.array([0.5] * 100, dtype=np.float32)

    # inconsistency with the shape
    desc = "Expected `context.shape[0]"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = NNPolicyLearner(
            n_actions=2, dim_context=2, off_policy_objective="ipw"
        )
        variant_context = np.ones((101, 2), dtype=np.float32)
        learner.fit(
            context=variant_context, action=action, reward=reward, pscore=pscore
        )

    # inconsistency between dim_context and context
    desc = "Expected `context.shape[1]"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = NNPolicyLearner(
            n_actions=2, dim_context=3, off_policy_objective="ipw"
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)


def test_nn_policy_learner_predict():
    n_actions = 2
    len_list = 1
    context = np.ones((100, 2), dtype=np.float32)
    context_test = np.array([i for i in range(10)], dtype=np.float32).reshape(5, 2)
    action = np.zeros((100,), dtype=int)
    reward = np.ones((100,), dtype=np.float32)
    pscore = np.array([0.5] * 100, dtype=np.float32)

    # shape error
    desc = "`context` must be 2D array"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective="ipw",
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([1.0, 1.0], dtype=np.float32)
        learner.predict(context=invalid_context)

    # inconsistency between dim_context and context
    desc = "Expected `context.shape[1]"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective="ipw",
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        learner.predict(context=invalid_context)

    # shape consistency of action_dist
    # n_rounds is 5, dim_context is 2
    learner = NNPolicyLearner(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=2,
        off_policy_objective="ipw",
    )
    learner.fit(context=context, action=action, reward=reward, pscore=pscore)
    action_dist = learner.predict(context=context_test)
    assert np.allclose(
        action_dist.sum(1), np.ones_like((context_test.shape[0], len_list))
    )
    assert action_dist.shape[0] == 5
    assert action_dist.shape[1] == n_actions
    assert action_dist.shape[2] == len_list


def test_nn_policy_learner_sample_action():
    n_actions = 2
    len_list = 1
    context = np.ones((100, 2), dtype=np.float32)
    context_test = np.array([i for i in range(10)], dtype=np.float32).reshape(5, 2)
    action = np.zeros((100,), dtype=int)
    reward = np.ones((100,), dtype=np.float32)
    pscore = np.array([0.5] * 100, dtype=np.float32)

    # shape error
    desc = "`context` must be 2D array"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective="ipw",
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([1.0, 1.0], dtype=np.float32)
        learner.sample_action(context=invalid_context)

    # inconsistency between dim_context and context
    desc = "Expected `context.shape[1]"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective="ipw",
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        learner.sample_action(context=invalid_context)

    learner = NNPolicyLearner(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=2,
        off_policy_objective="ipw",
    )
    learner.fit(context=context, action=action, reward=reward, pscore=pscore)
    action_dist = learner.sample_action(context=context_test)
    assert np.allclose(
        action_dist.sum(1), np.ones_like((context_test.shape[0], len_list))
    )
    assert action_dist.shape[0] == context_test.shape[0]
    assert action_dist.shape[1] == n_actions
    assert action_dist.shape[2] == len_list


def test_nn_policy_learner_predict_proba():
    n_actions = 2
    len_list = 1
    context = np.ones((100, 2), dtype=np.float32)
    context_test = np.array([i for i in range(10)], dtype=np.float32).reshape(5, 2)
    action = np.zeros((100,), dtype=int)
    reward = np.ones((100,), dtype=np.float32)
    pscore = np.array([0.5] * 100, dtype=np.float32)

    # shape error
    desc = "`context` must be 2D array"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective="ipw",
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([1.0, 1.0], dtype=np.float32)
        learner.predict_proba(context=invalid_context)

    # inconsistency between dim_context and context
    desc = "Expected `context.shape[1]"
    with pytest.raises(ValueError, match=f"{desc}*"):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective="ipw",
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        learner.predict_proba(context=invalid_context)

    learner = NNPolicyLearner(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=2,
        off_policy_objective="ipw",
    )
    learner.fit(context=context, action=action, reward=reward, pscore=pscore)
    action_dist = learner.predict_proba(context=context_test)
    assert np.allclose(
        action_dist.sum(1), np.ones_like((context_test.shape[0], len_list))
    )
    assert action_dist.shape[0] == context_test.shape[0]
    assert action_dist.shape[1] == n_actions
    assert action_dist.shape[2] == len_list
