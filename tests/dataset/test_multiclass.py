from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

from obp.dataset import MultiClassToBanditReduction


@pytest.fixture(scope="session")
def raw_data() -> Tuple[np.ndarray, np.ndarray]:
    X, y = load_digits(return_X_y=True)
    return X, y


def test_invalid_initialization(raw_data):
    X, y = raw_data

    # invalid alpha_b
    with pytest.raises(ValueError):
        MultiClassToBanditReduction(
            X=X, y=y, base_classifier_b=LogisticRegression(max_iter=10000), alpha_b=-0.3
        )

    with pytest.raises(ValueError):
        MultiClassToBanditReduction(
            X=X, y=y, base_classifier_b=LogisticRegression(max_iter=10000), alpha_b=1.3
        )

    # invalid classifier
    with pytest.raises(ValueError):
        from sklearn.tree import DecisionTreeRegressor

        MultiClassToBanditReduction(X=X, y=y, base_classifier_b=DecisionTreeRegressor)

    # invalid n_def_actions
    with pytest.raises(TypeError):
        MultiClassToBanditReduction(
            X=X,
            y=y,
            base_classifier_b=LogisticRegression(max_iter=10000),
            n_deficient_actions="aaa",
        )

    with pytest.raises(TypeError):
        MultiClassToBanditReduction(
            X=X,
            y=y,
            base_classifier_b=LogisticRegression(max_iter=10000),
            n_deficient_actions=None,
        )

    with pytest.raises(ValueError):
        MultiClassToBanditReduction(
            X=X,
            y=y,
            base_classifier_b=LogisticRegression(max_iter=10000),
            n_deficient_actions=-1,
        )

    with pytest.raises(ValueError):
        MultiClassToBanditReduction(
            X=X,
            y=y,
            base_classifier_b=LogisticRegression(max_iter=10000),
            n_deficient_actions=1000,
        )


def test_split_train_eval(raw_data):
    X, y = raw_data

    eval_size = 1000
    mcbr = MultiClassToBanditReduction(
        X=X, y=y, base_classifier_b=LogisticRegression(max_iter=10000), alpha_b=0.3
    )
    mcbr.split_train_eval(eval_size=eval_size)

    assert eval_size == mcbr.n_rounds_ev


def test_obtain_batch_bandit_feedback(raw_data):
    X, y = raw_data

    for n_deficient_actions in [0, 2]:
        mcbr = MultiClassToBanditReduction(
            X=X,
            y=y,
            base_classifier_b=LogisticRegression(max_iter=10000),
            alpha_b=0.3,
            n_deficient_actions=n_deficient_actions,
        )
        mcbr.split_train_eval()
        bandit_feedback = mcbr.obtain_batch_bandit_feedback()

        assert "n_actions" in bandit_feedback.keys()
        assert "n_rounds" in bandit_feedback.keys()
        assert "context" in bandit_feedback.keys()
        assert "action" in bandit_feedback.keys()
        assert "reward" in bandit_feedback.keys()
        assert "position" in bandit_feedback.keys()
        assert "pi_b" in bandit_feedback.keys()
        assert "pscore" in bandit_feedback.keys()

        n_rounds = bandit_feedback["n_rounds"]
        pi_b = bandit_feedback["pi_b"]
        assert pi_b.shape[0] == n_rounds
        n_actions = np.unique(y).shape[0]
        assert pi_b.shape[1] == n_actions
        assert pi_b.shape[2] == 1
        assert np.allclose(pi_b[:, :, 0].sum(1), np.ones(n_rounds))
        assert (pi_b == 0).sum() == n_deficient_actions * n_rounds


def test_obtain_action_dist_by_eval_policy(raw_data):
    X, y = raw_data

    eval_size = 1000
    mcbr = MultiClassToBanditReduction(
        X=X, y=y, base_classifier_b=LogisticRegression(max_iter=10000), alpha_b=0.3
    )
    mcbr.split_train_eval(eval_size=eval_size)

    # invalid alpha_e
    with pytest.raises(ValueError):
        mcbr.obtain_action_dist_by_eval_policy(alpha_e=-0.3)

    with pytest.raises(ValueError):
        mcbr.obtain_action_dist_by_eval_policy(alpha_e=1.3)

    # valid type
    action_dist = mcbr.obtain_action_dist_by_eval_policy()

    assert action_dist.shape[0] == eval_size
    n_actions = np.unique(y).shape[0]
    assert action_dist.shape[1] == n_actions
    assert action_dist.shape[2] == 1
    assert np.allclose(action_dist[:, :, 0].sum(1), np.ones(eval_size))


def test_calc_ground_truth_policy_value(raw_data):
    X, y = raw_data

    eval_size = 1000
    mcbr = MultiClassToBanditReduction(
        X=X, y=y, base_classifier_b=LogisticRegression(max_iter=10000), alpha_b=0.3
    )
    mcbr.split_train_eval(eval_size=eval_size)

    with pytest.raises(ValueError):
        invalid_action_dist = np.zeros(eval_size)
        mcbr.calc_ground_truth_policy_value(action_dist=invalid_action_dist)

    with pytest.raises(ValueError):
        reshaped_action_dist = mcbr.obtain_action_dist_by_eval_policy().reshape(
            1, -1, 1
        )
        mcbr.calc_ground_truth_policy_value(action_dist=reshaped_action_dist)

    action_dist = mcbr.obtain_action_dist_by_eval_policy()
    ground_truth_policy_value = mcbr.calc_ground_truth_policy_value(
        action_dist=action_dist
    )
    assert isinstance(ground_truth_policy_value, float)
