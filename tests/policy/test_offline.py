import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

from obp.policy.offline import IPWLearner
from obp.policy.offline import NNPolicyLearner
from obp.policy.policy_type import PolicyType
from obp.ope.estimators import InverseProbabilityWeighting


def test_base_opl_init():
    # n_actions
    with pytest.raises(ValueError):
        IPWLearner(n_actions=1)

    with pytest.raises(ValueError):
        IPWLearner(n_actions="3")

    # len_list
    with pytest.raises(ValueError):
        IPWLearner(n_actions=2, len_list=0)

    with pytest.raises(ValueError):
        IPWLearner(n_actions=2, len_list="3")

    # policy_type
    assert IPWLearner(n_actions=2).policy_type == PolicyType.OFFLINE

    # invalid relationship between n_actions and len_list
    with pytest.raises(ValueError):
        IPWLearner(n_actions=5, len_list=10)

    with pytest.raises(ValueError):
        IPWLearner(n_actions=2, len_list=3)


def test_ipw_learner_init():
    # base classifier
    len_list = 2
    learner1 = IPWLearner(n_actions=2, len_list=len_list)
    assert isinstance(learner1.base_classifier, LogisticRegression)
    for i in range(len_list):
        assert isinstance(learner1.base_classifier_list[i], LogisticRegression)

    with pytest.raises(ValueError):
        from sklearn.linear_model import LinearRegression

        IPWLearner(n_actions=2, base_classifier=LinearRegression())

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
    context = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, -1)
    action = np.array([0, 1])
    reward = np.array([1.0, 0.0])
    position = np.array([0, 0])
    learner = IPWLearner(n_actions=2, len_list=1)
    learner.fit(context=context, action=action, reward=reward, position=position)

    # inconsistency with the shape
    with pytest.raises(ValueError):
        learner = IPWLearner(n_actions=2, len_list=2)
        variant_context = np.array([1.0, 1.0, 1.0, 1.0])
        learner.fit(
            context=variant_context, action=action, reward=reward, position=position
        )

    # len_list > 2, but position is not set
    with pytest.raises(ValueError):
        learner = IPWLearner(n_actions=2, len_list=2)
        learner.fit(context=context, action=action, reward=reward)


def test_ipw_learner_predict():
    n_actions = 2
    len_list = 1

    # shape error
    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
        invalid_type_context = [1.0, 2.0]
        learner.sample_action(context=invalid_type_context)

    with pytest.raises(ValueError):
        invalid_ndim_context = np.array([1.0, 2.0, 3.0, 4.0])
        learner.sample_action(context=invalid_ndim_context)

    context = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, -1)
    n_rounds = context.shape[0]
    sampled_action = learner.sample_action(context=context)

    assert sampled_action.shape[0] == n_rounds
    assert sampled_action.shape[1] == n_actions
    assert sampled_action.shape[2] == len_list


ipw = InverseProbabilityWeighting()

# n_actions, len_list, dim_context, off_policy_objective, hidden_layer_size, activation, solver, alpha,
# batch_size, learning_rate_init, max_iter, shuffle, random_state, tol, momentum, nesterovs_momentum,
# early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun, description
invalid_input_of_nn_policy_learner_init = [
    (
        0,  #
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "n_actions must be an integer larger than 1",
    ),
    (
        10,
        -1,  #
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "len_list must be a positive integer",
    ),
    (
        10,
        1,
        -1,  #
        ipw.estimate_policy_value_tensor,
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
        15000,
        "dim_context must be a positive integer",
    ),
    (
        10,
        1,
        2,
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
        15000,
        "off_policy_objective must be callable",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "hidden_layer_size must be tuple of positive integers",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "activation must be one of 'identity', 'logistic', 'tanh', or 'relu'",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "solver must be one of 'adam', 'lbfgs', or 'sgd'",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
        (100, 50, 100),
        "relu",
        "adam",
        -1,  #
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
        15000,
        "alpha must be a non-negative float",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "batch_size must be a positive integer or 'auto'",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0,  #
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
        15000,
        "learning_rate_init must be a positive float",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "max_iter must be a positive integer",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "shuffle must be a bool",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "'' cannot be used to seed",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
        (100, 50, 100),
        "relu",
        "adam",
        0.001,
        "auto",
        0.0001,
        200,
        True,
        123,
        -1,  #
        0.9,
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        15000,
        "tol must be a positive float",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        2,  #
        True,
        True,
        0.1,
        0.9,
        0.999,
        1e-8,
        10,
        15000,
        "momentum must be a float in [0., 1.]",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "nesterovs_momentum must be a bool",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "early_stopping must be a bool",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "if early_stopping is True, solver must be one of 'sgd' or 'adam'",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        2,  #
        0.9,
        0.999,
        1e-8,
        10,
        15000,
        "validation_fraction must be a float in",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        2,  #
        0.999,
        1e-8,
        10,
        15000,
        "beta_1 must be a float in [0. 1.]",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        2,  #
        1e-8,
        10,
        15000,
        "beta_2 must be a float in [0., 1.]",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        -1,  #
        10,
        15000,
        "epsilon must be a non-negative float",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "n_iter_no_change must be a positive integer",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        0,  #
        "max_fun must be a positive integer",
    ),
]

valid_input_of_nn_policy_learner_init = [
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
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
        15000,
        "valid input",
    ),
    (
        10,
        1,
        2,
        ipw.estimate_policy_value_tensor,
        (100, 50, 100),
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
        15000,
        "valid input",
    ),
]


@pytest.mark.parametrize(
    "n_actions, len_list, dim_context, off_policy_objective, hidden_layer_size, activation, solver, alpha, batch_size, learning_rate_init, max_iter, shuffle, random_state, tol, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun, description",
    invalid_input_of_nn_policy_learner_init,
)
def test_nn_policy_learner_init_using_invalid_inputs(
    n_actions,
    len_list,
    dim_context,
    off_policy_objective,
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
    max_fun,
    description,
):
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=dim_context,
            off_policy_objective=off_policy_objective,
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
            max_fun=max_fun,
        )


@pytest.mark.parametrize(
    "n_actions, len_list, dim_context, off_policy_objective, hidden_layer_size, activation, solver, alpha, batch_size, learning_rate_init, max_iter, shuffle, random_state, tol, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun, description",
    valid_input_of_nn_policy_learner_init,
)
def test_nn_policy_learner_init_using_valid_inputs(
    n_actions,
    len_list,
    dim_context,
    off_policy_objective,
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
    max_fun,
    description,
):
    nn_policy_learner = NNPolicyLearner(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=dim_context,
        off_policy_objective=off_policy_objective,
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
        max_fun=max_fun,
    )
    assert isinstance(nn_policy_learner, NNPolicyLearner)


def test_nn_policy_learner_create_train_data_for_opl():
    context = np.ones((100, 2), dtype=np.int32)
    action = np.zeros((100,), dtype=np.int32)
    reward = np.ones((100,), dtype=np.float32)
    pscore = np.array([0.5] * 100, dtype=np.float32)
    estimated_rewards_by_reg_model = np.ones((100, 2), dtype=np.float32)
    position = np.zeros((100,), dtype=np.int32)
    ipw = InverseProbabilityWeighting()

    learner1 = NNPolicyLearner(
        n_actions=2,
        dim_context=2,
        off_policy_objective=ipw.estimate_policy_value_tensor,
    )

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
        off_policy_objective=ipw.estimate_policy_value_tensor,
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
    ipw = InverseProbabilityWeighting()

    # inconsistency with the shape
    with pytest.raises(ValueError):
        learner = NNPolicyLearner(
            n_actions=2,
            dim_context=2,
            off_policy_objective=ipw.estimate_policy_value_tensor,
        )
        variant_context = np.ones((101, 2), dtype=np.float32)
        learner.fit(
            context=variant_context, action=action, reward=reward, pscore=pscore
        )

    # inconsistency between dim_context and context
    with pytest.raises(ValueError):
        learner = NNPolicyLearner(
            n_actions=2,
            dim_context=3,
            off_policy_objective=ipw.estimate_policy_value_tensor,
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
    ipw = InverseProbabilityWeighting()

    # shape error
    with pytest.raises(ValueError):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective=ipw.estimate_policy_value_tensor,
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([1.0, 1.0], dtype=np.float32)
        learner.predict(context=invalid_context)

    # inconsistency between dim_context and context
    with pytest.raises(ValueError):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective=ipw.estimate_policy_value_tensor,
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
        off_policy_objective=ipw.estimate_policy_value_tensor,
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
    ipw = InverseProbabilityWeighting()

    # shape error
    with pytest.raises(ValueError):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective=ipw.estimate_policy_value_tensor,
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([1.0, 1.0], dtype=np.float32)
        learner.sample_action(context=invalid_context)

    # inconsistency between dim_context and context
    with pytest.raises(ValueError):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective=ipw.estimate_policy_value_tensor,
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        learner.sample_action(context=invalid_context)

    learner = NNPolicyLearner(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=2,
        off_policy_objective=ipw.estimate_policy_value_tensor,
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
    ipw = InverseProbabilityWeighting()

    # shape error
    with pytest.raises(ValueError):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective=ipw.estimate_policy_value_tensor,
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([1.0, 1.0], dtype=np.float32)
        learner.predict_proba(context=invalid_context)

    # inconsistency between dim_context and context
    with pytest.raises(ValueError):
        learner = NNPolicyLearner(
            n_actions=n_actions,
            len_list=len_list,
            dim_context=2,
            off_policy_objective=ipw.estimate_policy_value_tensor,
        )
        learner.fit(context=context, action=action, reward=reward, pscore=pscore)
        invalid_context = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        learner.predict_proba(context=invalid_context)

    learner = NNPolicyLearner(
        n_actions=n_actions,
        len_list=len_list,
        dim_context=2,
        off_policy_objective=ipw.estimate_policy_value_tensor,
    )
    learner.fit(context=context, action=action, reward=reward, pscore=pscore)
    action_dist = learner.predict_proba(context=context_test)
    assert np.allclose(
        action_dist.sum(1), np.ones_like((context_test.shape[0], len_list))
    )
    assert action_dist.shape[0] == context_test.shape[0]
    assert action_dist.shape[1] == n_actions
    assert action_dist.shape[2] == len_list
