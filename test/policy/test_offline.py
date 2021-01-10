import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression


from obp.policy.offline import IPWLearner


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
    assert IPWLearner(n_actions=2).policy_type == "offline"

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


def test_create_train_data_for_opl():
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


def test_opl_fit():
    context = np.array([1.0, 1.0, 1.0, 1.0]).reshape(2, -1)
    action = np.array([0, 1])
    reward = np.array([1.0, 0.0])
    position = np.array([0, 0])
    learner = IPWLearner(n_actions=2, len_list=1)
    learner.fit(context=context, action=action, reward=reward, position=position)

    # inconsistency with the shape
    with pytest.raises(AssertionError):
        learner = IPWLearner(n_actions=2, len_list=2)
        variant_context = np.array([1.0, 1.0, 1.0, 1.0])
        learner.fit(
            context=variant_context, action=action, reward=reward, position=position
        )

    # len_list > 2, but position is not set
    with pytest.raises(ValueError):
        learner = IPWLearner(n_actions=2, len_list=2)
        learner.fit(context=context, action=action, reward=reward)


def test_opl_predict():
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

    context_predict = np.array([i for i in range(10)]).reshape(5, 2)
    action_dist = learner.predict(context=context_predict)
    assert action_dist.shape[0] == 5
    assert action_dist.shape[1] == n_actions
    assert action_dist.shape[2] == len_list
