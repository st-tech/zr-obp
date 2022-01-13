from pathlib import Path
from typing import Dict

from conftest import generate_action_dist
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import yaml

from obp.ope import ImportanceWeightEstimator
from obp.types import BanditFeedback


np.random.seed(1)

binary_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=GradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

# hyperparameter settings for the base ML model in importance weight estimator
cd_path = Path(__file__).parent.resolve()
with open(cd_path / "hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)


# action_context, n_actions, len_list, fitting_method, base_model, calibration_cv, err, description
n_rounds = 1000
n_actions = 3
len_list = 3

invalid_input_of_initializing_importance_weight_estimator = [
    (
        np.random.uniform(size=(n_actions, 8)),
        "a",  #
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        2,
        TypeError,
        "n_actions must be an instance of <class 'int'>, not <class 'str'>.",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        1,  #
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        2,
        ValueError,
        "n_actions == 1, must be >= 2",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        "a",  #
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        2,
        TypeError,
        "len_list must be an instance of <class 'int'>, not <class 'str'>.",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        0,  #
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        2,
        ValueError,
        "len_list == 0, must be >= 1",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        1,  #
        RandomForestClassifier(**hyperparams["random_forest"]),
        2,
        ValueError,
        "`fitting_method` must be either 'sample' or 'raw', but 1 is given",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "awesome",  #
        RandomForestClassifier(**hyperparams["random_forest"]),
        2,
        ValueError,
        "`fitting_method` must be either 'sample' or 'raw', but awesome is given",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        "RandomForest",  #
        2,
        ValueError,
        "`base_model` must be BaseEstimator or a child class of BaseEstimator",
    ),
    (
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        1.5,
        TypeError,
        "calibration_cv must be an instance of <class 'int'>, not <class 'float'>.",
    ),
]


# context, action, position, action_context, n_actions, len_list, fitting_method, base_model, action_dist, n_folds, random_state, calibration_cv, err, description
invalid_input_of_fitting_importance_weight_estimator = [
    (
        None,  #
        np.random.choice(n_actions, size=n_rounds),
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`context` must be 2D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        None,  #
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`action` must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7, 3)),  #
        np.random.choice(n_actions, size=n_rounds),
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`context` must be 2D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=(n_rounds, 3)),  #
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`action` must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(["1", "a"], size=n_rounds),  #
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`action` elements must be non-negative integers",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice([-1, -3], size=n_rounds),  #
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`action` elements must be non-negative integers",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        "3",  #
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`position` must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.choice(len_list, size=(n_rounds, 3)),  #
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`position` must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.choice(len_list, size=n_rounds - 1),  #
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "Expected `context.shape[0]",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.choice(["a", "1"], size=n_rounds),  #
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`position` elements must be non-negative integers",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds),
        np.random.choice([-1, -3], size=n_rounds),  #
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`position` elements must be non-negative integers",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds - 1),  #
        None,
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "Expected `context.shape[0]",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_actions, size=n_rounds - 1),  #
        None,
        None,
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "Expected `context.shape[0]",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.choice(len_list, size=n_rounds),
        "3",  #
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`action_context` must be 2D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8, 3)),  #
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`action_context` must be 2D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        (np.arange(n_rounds) % n_actions) + 1,  #
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        r"`action` elements must be integers in the range of",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.ones((n_rounds, 2)),  #
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`position` must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.ones(n_rounds, dtype=int) * len_list,  #
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "`position` elements must be smaller than `len_list`",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        None,  #
        3,
        1,
        2,
        ValueError,
        "`action_dist` must be 3D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        np.zeros((n_rounds, n_actions, len_list - 1)),  #
        3,
        1,
        2,
        ValueError,
        "shape of `action_dist` must be (n_rounds, n_actions, len_list)",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        np.zeros((n_rounds, n_actions, len_list)),  #
        3,
        1,
        2,
        ValueError,
        "`action_dist` must be a probability distribution",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        "a",  #
        None,
        2,
        TypeError,
        "n_folds must be an instance of <class 'int'>, not <class 'str'>",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        0,  #
        None,
        2,
        ValueError,
        "n_folds == 0, must be >= 1.",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        "a",  #
        2,
        ValueError,
        "'a' cannot be used to seed a numpy.random.RandomState instance",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.zeros(n_rounds, dtype=int),  #
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        3,
        1,
        2,
        ValueError,
        "No training data at position",
    ),
]


valid_input_of_importance_weight_estimator = [
    (
        np.random.uniform(size=(n_rounds * 100, 7)),
        np.arange(n_rounds * 100) % n_actions,
        np.random.choice(len_list, size=n_rounds * 100),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds * 100, n_actions, len_list),
        3,
        1,
        2,
        "valid input with cross fitting",
    ),
    (
        np.random.uniform(size=(n_rounds * 100, 7)),
        np.arange(n_rounds * 100) % n_actions,
        np.random.choice(len_list, size=n_rounds * 100),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds * 100, n_actions, len_list),
        3,
        2,
        1,
        "valid input with cross fitting",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.choice(len_list, size=n_rounds),
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        len_list,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        1,
        1,
        2,
        "valid input without cross fitting",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        None,
        np.random.uniform(size=(n_actions, 8)),
        n_actions,
        1,
        "sample",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, 1),
        1,
        1,
        2,
        "valid input without position",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        None,
        None,
        n_actions,
        1,
        "raw",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, 1),
        1,
        1,
        2,
        "valid input without position when fitting_method is `raw`",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.arange(n_rounds) % n_actions,
        np.random.choice(len_list, size=n_rounds),
        None,
        n_actions,
        len_list,
        "raw",
        RandomForestClassifier(**hyperparams["random_forest"]),
        generate_action_dist(n_rounds, n_actions, len_list),
        1,
        1,
        2,
        "valid input when fitting_method is `raw`",
    ),
]


@pytest.mark.parametrize(
    "action_context, n_actions, len_list, fitting_method, base_model, calibration_cv, err, description",
    invalid_input_of_initializing_importance_weight_estimator,
)
def test_initializing_importance_weight_estimator_using_invalid_input_data(
    action_context: np.ndarray,
    n_actions: int,
    len_list: int,
    fitting_method: str,
    base_model: BaseEstimator,
    calibration_cv: int,
    err,
    description: str,
) -> None:
    # initialization raises ValueError
    with pytest.raises(err, match=f"{description}*"):
        _ = ImportanceWeightEstimator(
            n_actions=n_actions,
            len_list=len_list,
            action_context=action_context,
            base_model=base_model,
            fitting_method=fitting_method,
            calibration_cv=calibration_cv,
        )


@pytest.mark.parametrize(
    "context, action, position, action_context, n_actions, len_list, fitting_method, base_model, action_dist, n_folds, random_state, calibration_cv, err, description",
    invalid_input_of_fitting_importance_weight_estimator,
)
def test_fitting_importance_weight_estimator_using_invalid_input_data(
    context: np.ndarray,
    action: np.ndarray,
    position: np.ndarray,
    action_context: np.ndarray,
    n_actions: int,
    len_list: int,
    fitting_method: str,
    base_model: BaseEstimator,
    action_dist: np.ndarray,
    n_folds: int,
    random_state: int,
    calibration_cv: int,
    err,
    description: str,
) -> None:
    # fit_predict function raises ValueError
    with pytest.raises(err, match=f"{description}*"):
        importance_weight_estimator = ImportanceWeightEstimator(
            n_actions=n_actions,
            len_list=len_list,
            action_context=action_context,
            base_model=base_model,
            fitting_method=fitting_method,
            calibration_cv=calibration_cv,
        )
        # train importance weight estimator on logged bandit feedback data
        _ = importance_weight_estimator.fit_predict(
            context=context,
            action=action,
            position=position,
            n_folds=n_folds,
            random_state=random_state,
            action_dist=action_dist,
        )


@pytest.mark.parametrize(
    "context, action, position, action_context, n_actions, len_list, fitting_method, base_model, action_dist, n_folds, random_state, calibration_cv, description",
    valid_input_of_importance_weight_estimator,
)
def test_importance_weight_estimator_using_valid_input_data(
    context: np.ndarray,
    action: np.ndarray,
    position: np.ndarray,
    action_context: np.ndarray,
    n_actions: int,
    len_list: int,
    fitting_method: str,
    base_model: BaseEstimator,
    action_dist: np.ndarray,
    n_folds: int,
    random_state: int,
    calibration_cv: int,
    description: str,
) -> None:
    # fit_predict
    importance_weight_estimator = ImportanceWeightEstimator(
        n_actions=n_actions,
        len_list=len_list,
        action_context=action_context,
        base_model=base_model,
        fitting_method=fitting_method,
        calibration_cv=calibration_cv,
    )
    # train importance weight estimator on logged bandit feedback data
    _ = importance_weight_estimator.fit_predict(
        context=context,
        action=action,
        action_dist=action_dist,
        position=position,
        n_folds=n_folds,
        random_state=random_state,
    )


def test_performance_of_binary_outcome_models(
    fixed_synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the performance of ope estimators using synthetic bandit data and random evaluation policy
    when the importance weight estimator is estimated by a logistic regression
    """
    bandit_feedback = fixed_synthetic_bandit_feedback.copy()
    action_dist = random_action_dist
    random_state = 12345
    auc_scores: Dict[str, float] = {}
    fit_methods = ["sample", "raw"]
    for fit_method in fit_methods:
        for model_name, model in binary_model_dict.items():
            importance_weight_estimator = ImportanceWeightEstimator(
                n_actions=bandit_feedback["n_actions"],
                action_context=bandit_feedback["action_context"],
                base_model=model(**hyperparams[model_name]),
                fitting_method=fit_method,
                len_list=1,
            )
            # train importance weight estimator on logged bandit feedback data
            estimated_importance_weight = importance_weight_estimator.fit_predict(
                context=bandit_feedback["context"],
                action=bandit_feedback["action"],
                action_dist=action_dist,
                n_folds=2,  # 2-fold cross-fitting
                random_state=random_state,
                evaluate_model_performance=True,
            )
            assert np.all(
                estimated_importance_weight >= 0
            ), "estimated_importance_weight must be non-negative"
            # extract predictions
            tmp_y = []
            tmp_pred = []
            for i in range(len(importance_weight_estimator.eval_result["y"])):
                tmp_y.append(importance_weight_estimator.eval_result["y"][i])
                tmp_pred.append(importance_weight_estimator.eval_result["proba"][i])
            y_test = np.array(tmp_y).flatten()
            y_pred = np.array(tmp_pred).flatten()
            auc_scores[model_name + "_" + fit_method] = roc_auc_score(
                y_true=y_test,
                y_score=y_pred,
            )

    for model_name in auc_scores:
        print(f"AUC of {model_name} is {auc_scores[model_name]}")
        assert (
            auc_scores[model_name] > 0.5
        ), f"AUC of {model_name} should be greater than 0.5"
