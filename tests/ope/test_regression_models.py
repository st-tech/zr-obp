from typing import Dict
from pathlib import Path
import yaml

import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
import pytest

from obp.ope import RegressionModel
from obp.types import BanditFeedback
from conftest import generate_action_dist


np.random.seed(1)

binary_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

# hyperparameter settings for the base ML model in regression model
cd_path = Path(__file__).parent.resolve()
with open(cd_path / "hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)


# action_context, n_actions, len_list, fitting_method, base_model, description
n_rounds = 1000
n_actions = 3
len_list = 3

invalid_input_of_initializing_regression_models = [
    (
        np.random.uniform(size=(n_actions, 8)),
        "a",  #
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        "n_actions must be an integer larger than 1",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        1,  #
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        "n_actions must be an integer larger than 1",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        "a",  #
        "normal",
        Ridge(**hyperparams["ridge"]),
        "len_list must be a positive integer",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        0,  #
        "normal",
        Ridge(**hyperparams["ridge"]),
        "len_list must be a positive integer",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        1,  #
        Ridge(**hyperparams["ridge"]),
        "fitting_method must be one of",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "awesome",  #
        Ridge(**hyperparams["ridge"]),
        "fitting_method must be one of",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "normal",
        "RandomForest",  #
        "base_model must be BaseEstimator or a child class of BaseEstimator",
    ),
]


# context, action, reward, pscore, position, action_context, n_actions, len_list, fitting_method, base_model, action_dist, description
invalid_input_of_fitting_regression_models = [
    (
        None,  #
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "context must be ndarray",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        None,  #
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "action must be ndarray",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        None,  #
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "reward must be ndarray",
    ),
    (
        np.random.uniform(size=(n_rounds, 7, 3)),  #
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "context must be 2-dimensional",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=(n_rounds, 3)),  #
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "action must be 1-dimensional",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=(n_rounds, 3)),  #
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "reward must be 1-dimensional",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(["1", "a"], size=n_rounds),  #
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "action elements must be non-negative integers",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice([-1, -3], size=n_rounds),  #
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "action elements must be non-negative integers",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        "3",  #
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "iw",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        "pscore must be ndarray",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        np.ones((n_rounds, 2)) * 2,  #
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "iw",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        "pscore must be 1-dimensional",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds - 1) * 2,  #
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "iw",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        "context, action, reward, and pscore must be the same size.",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        np.arange(n_rounds),  #
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "iw",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        "pscore must be positive",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        "3",  #
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "position must be ndarray",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=(n_rounds, 3)),  #
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "position must be 1-dimensional",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds - 1),  #
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "context, action, reward, and position must be the same size.",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(["a", "1"], size=n_rounds),  #
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "position elements must be non-negative integers",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice([-1, -3], size=n_rounds),  #
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "position elements must be non-negative integers",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds - 1),  #
        np.random.uniform(size=n_rounds),
        None,
        None,
        None,
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "context, action, and reward must be the same size",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds - 1),  #
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        None,
        None,
        n_actions,
        len_list,
        "iw",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        "context, action, reward, and pscore must be the same size",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        "3",  #
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "action_context must be ndarray",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8, 3)),  #
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "action_context must be 2-dimensional",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        (np.arange(n_rounds) % n_actions) + 1,  #
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "action elements must be smaller than the size of the first dimension of action_context",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.ones((n_rounds, 2)),  #
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "position must be 1-dimensional",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.ones(n_rounds, dtype=int) * len_list,  #
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        3,
        1,
        "position elements must be smaller than len_list",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "iw",
        Ridge(**hyperparams["ridge"]),
        None,  #
        3,
        1,
        "when fitting_method is either",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "mrdr",
        Ridge(**hyperparams["ridge"]),
        None,  #
        3,
        1,
        "when fitting_method is either",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "iw",
        Ridge(**hyperparams["ridge"]),
        np.zeros((n_rounds, n_actions, len_list - 1)),  #
        3,
        1,
        "shape of action_dist must be (n_rounds, n_actions, len_list)",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "iw",
        Ridge(**hyperparams["ridge"]),
        np.zeros((n_rounds, n_actions, len_list)),  #
        3,
        1,
        "action_dist must be a probability distribution",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        0,  #
        None,
        "n_folds must be a positive integer",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        "a",  #
        None,
        "n_folds must be a positive integer",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        "a",  #
        "random_state must be an integer",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.zeros(n_rounds, dtype=int),  #
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        "No training data at position",
    ),
]


valid_input_of_regression_models = [
    (
        np.random.uniform(size=(n_rounds * 100, 7)),
        np.arange(n_rounds * 100) % n_actions,
        np.random.uniform(size=n_rounds * 100),
        np.ones(n_rounds * 100) * 2,
        np.random.choice(len_list, size=n_rounds * 100),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds * 100, n_actions, len_list),
        3,
        1,
        "valid input with cross fitting",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        1,
        1,
        "valid input without cross fitting",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        None,
        None,
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        1,
        "normal",
        Ridge(**hyperparams["ridge"]),
        None,
        1,
        1,
        "valid input without pscore, position, and action_dist",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.uniform(size=n_rounds),
        np.ones(n_rounds) * 2,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "iw",
        Ridge(**hyperparams["ridge"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        1,
        1,
        "valid input when fitting_method is iw",
    ),
]


@pytest.mark.parametrize(
    "action_context, n_actions, len_list, fitting_method, base_model, description",
    invalid_input_of_initializing_regression_models,
)
def test_initializing_regression_models_using_invalid_input_data(
    action_context: np.ndarray,
    n_actions: int,
    len_list: int,
    fitting_method: str,
    base_model: BaseEstimator,
    description: str,
) -> None:
    # initialization raises ValueError
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = RegressionModel(
            n_actions=n_actions,
            len_list=len_list,
            action_context=action_context,
            base_model=base_model,
            fitting_method=fitting_method,
        )


@pytest.mark.parametrize(
    "context, action, reward, pscore, position, action_context, n_actions, len_list, fitting_method, base_model, action_dist, n_folds, random_state, description",
    invalid_input_of_fitting_regression_models,
)
def test_fitting_regression_models_using_invalid_input_data(
    context: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    action_context: np.ndarray,
    n_actions: int,
    len_list: int,
    fitting_method: str,
    base_model: BaseEstimator,
    action_dist: np.ndarray,
    n_folds: int,
    random_state: int,
    description: str,
) -> None:
    # fit_predict function raises ValueError
    with pytest.raises(ValueError, match=f"{description}*"):
        regression_model = RegressionModel(
            n_actions=n_actions,
            len_list=len_list,
            action_context=action_context,
            base_model=base_model,
            fitting_method=fitting_method,
        )
        if fitting_method == "normal":
            # train regression model on logged bandit feedback data
            _ = regression_model.fit_predict(
                context=context,
                action=action,
                reward=reward,
                position=position,
                n_folds=n_folds,
                random_state=random_state,
            )
        else:
            # train regression model on logged bandit feedback data
            _ = regression_model.fit_predict(
                context=context,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                action_dist=action_dist,
                n_folds=n_folds,
                random_state=random_state,
            )


@pytest.mark.parametrize(
    "context, action, reward, pscore, position, action_context, n_actions, len_list, fitting_method, base_model, action_dist, n_folds, random_state, description",
    valid_input_of_regression_models,
)
def test_regression_models_using_valid_input_data(
    context: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    action_context: np.ndarray,
    n_actions: int,
    len_list: int,
    fitting_method: str,
    base_model: BaseEstimator,
    action_dist: np.ndarray,
    n_folds: int,
    random_state: int,
    description: str,
) -> None:
    # fit_predict
    regression_model = RegressionModel(
        n_actions=n_actions,
        len_list=len_list,
        action_context=action_context,
        base_model=base_model,
        fitting_method=fitting_method,
    )
    if fitting_method == "normal":
        # train regression model on logged bandit feedback data
        _ = regression_model.fit_predict(
            context=context,
            action=action,
            reward=reward,
            position=position,
            n_folds=n_folds,
            random_state=random_state,
        )
    else:
        # train regression model on logged bandit feedback data
        _ = regression_model.fit_predict(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            action_dist=action_dist,
            n_folds=n_folds,
            random_state=random_state,
        )


def test_performance_of_binary_outcome_models(
    fixed_synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the performance of ope estimators using synthetic bandit data and random evaluation policy
    when the regression model is estimated by a logistic regression
    """
    bandit_feedback = fixed_synthetic_bandit_feedback.copy()
    expected_reward = bandit_feedback["expected_reward"][:, :, np.newaxis]
    action_dist = random_action_dist
    # compute ground truth policy value using expected reward
    q_pi_e = np.average(expected_reward[:, :, 0], weights=action_dist[:, :, 0], axis=1)
    # compute statistics of ground truth policy value
    gt_mean = q_pi_e.mean()
    random_state = 12345
    auc_scores: Dict[str, float] = {}
    # check ground truth
    print(f"gt_mean: {gt_mean}")
    # check the performance of regression models using doubly robust criteria (|\hat{q} - q| <= |q| is satisfied with a high probability)
    dr_criteria_pass_rate = 0.8
    fit_methods = ["normal", "iw", "mrdr"]
    for fit_method in fit_methods:
        for model_name, model in binary_model_dict.items():
            regression_model = RegressionModel(
                n_actions=bandit_feedback["n_actions"],
                action_context=bandit_feedback["action_context"],
                base_model=model(**hyperparams[model_name]),
                fitting_method=fit_method,
            )
            if fit_method == "normal":
                # train regression model on logged bandit feedback data
                estimated_rewards_by_reg_model = regression_model.fit_predict(
                    context=bandit_feedback["context"],
                    action=bandit_feedback["action"],
                    reward=bandit_feedback["reward"],
                    n_folds=3,  # 3-fold cross-fitting
                    random_state=random_state,
                )
            else:
                # train regression model on logged bandit feedback data
                estimated_rewards_by_reg_model = regression_model.fit_predict(
                    context=bandit_feedback["context"],
                    action=bandit_feedback["action"],
                    reward=bandit_feedback["reward"],
                    pscore=bandit_feedback["pscore"],
                    action_dist=action_dist,
                    n_folds=3,  # 3-fold cross-fitting
                    random_state=random_state,
                )
            auc_scores[model_name + "_" + fit_method] = roc_auc_score(
                y_true=bandit_feedback["reward"],
                y_score=estimated_rewards_by_reg_model[
                    np.arange(bandit_feedback["reward"].shape[0]),
                    bandit_feedback["action"],
                    np.zeros_like(bandit_feedback["action"], dtype=int),
                ],
            )
            # compare dr criteria
            dr_criteria = np.abs((gt_mean - estimated_rewards_by_reg_model)) - np.abs(
                gt_mean
            )
            print(
                f"Dr criteria is satisfied with probability {np.mean(dr_criteria <= 0)} ------ model: {model_name} ({fit_method}),"
            )
            assert (
                np.mean(dr_criteria <= 0) >= dr_criteria_pass_rate
            ), f" should be satisfied with a probability at least {dr_criteria_pass_rate}"

    for model_name in auc_scores:
        print(f"AUC of {model_name} is {auc_scores[model_name]}")
        assert (
            auc_scores[model_name] > 0.5
        ), f"AUC of {model_name} should be greater than 0.5"
