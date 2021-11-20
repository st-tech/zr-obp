from pathlib import Path
from typing import Dict

from itertools import permutations
from itertools import product
import numpy as np
import pytest
import yaml

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from obp.dataset import linear_behavior_policy_logit
from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticSlateBanditDataset
from obp.ope import SlateRewardInteractionIPS
from obp.ope import SlateCascadeDoublyRobust
from obp.ope import SlateRegressionModel
from obp.utils import softmax



np.random.seed(1)

model_dict = dict(
    linear_regression=LinearRegression,
    lightgbm=GradientBoostingRegressor,
    random_forest=RandomForestRegressor,
)

# hyperparameter settings for the base ML model in regression model
cd_path = Path(__file__).parent.resolve()
with open(cd_path / "hyperparams_slate.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)


n_rounds = 1000
n_unique_action = 3
len_list = 3
rips = SlateRewardInteractionIPS(len_list=len_list)
dr = SlateCascadeDoublyRobust(len_list=len_list, n_unique_action=n_unique_action)

# n_unique_action, len_list, fitting_method, base_model, err, description
invalid_input_of_initializing_regression_models = [
    (
        "a",  #
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        TypeError,
        "`n_unique_action` must be an instance of <class 'int'>, not <class 'str'>.",
    ),
    (
        1,  #
        len_list,
        "normal",
        Ridge(**hyperparams["ridge"]),
        ValueError,
        "`n_unique_action`= 1, must be >= 2",
    ),
    (
        n_unique_action,
        "a",  #
        "normal",
        Ridge(**hyperparams["ridge"]),
        TypeError,
        "`len_list` must be an instance of <class 'int'>, not <class 'str'>.",
    ),
    (
        n_unique_action,
        0,  #
        "normal",
        Ridge(**hyperparams["ridge"]),
        ValueError,
        "`len_list`= 0, must be >= 1",
    ),
    (
        n_unique_action,
        len_list,
        1,  #
        Ridge(**hyperparams["ridge"]),
        ValueError,
        "fitting_method must be one of",
    ),
    (
        n_unique_action,
        len_list,
        "awesome",  #
        Ridge(**hyperparams["ridge"]),
        ValueError,
        "fitting_method must be one of",
    ),
    (
        n_unique_action,
        len_list,
        "normal",
        "RandomForest",  #
        ValueError,
        "base_model must be BaseEstimator or a child class of BaseEstimator",
    ),
]


# context, action, reward, pscore, evaluation_policy_pscore, evaluation_policy_action_dist, err, description
invalid_input_of_fitting_regression_models = [
    (
        None,  #
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "context must be 2D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7, 3)),  #
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "context must be 2D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        None,  #
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "action must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=(n_rounds, len_list)),  #
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "action must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(["1", "a"], size=n_rounds * len_list),  #
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "action elements must be non-negative integers",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice([-1, -3], size=n_rounds * len_list),  #
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "action elements must be non-negative integers",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        (np.arange(n_rounds) % n_unique_action) + 1,  #
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "action elements must be smaller than",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list - 1),  #
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "Expected `context.shape[0]",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        None,  #
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "reward must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=(n_rounds, len_list)),  #
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "reward must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        "3",  #
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "pscore must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones((n_rounds, 2)) * 2,  #
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "pscore must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(n_rounds - 1) * 2,  #
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "Expected `context.shape[0]",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.arange(n_rounds),  #
        np.random.uniform(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "pscore must be positive",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        "3",  #
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "evaluation_policy_pscore must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones((n_rounds, 2)) * 2,  #
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "evaluation_policy_pscore must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(n_rounds - 1) * 2,  #
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "Expected `context.shape[0]",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.arange(n_rounds),  #
        np.ones(size=n_rounds * len_list * n_unique_action),
        ValueError,
        "evaluation_policy_pscore must be positive",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        "4",  #
        "evaluation_policy_action_dist must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones((n_rounds, len_list, n_unique_action)) / n_unique_action,  #
        "evaluation_policy_action_dist must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones((n_rounds * len_list, n_unique_action)) / n_unique_action,  #
        "evaluation_policy_action_dist must be 1D array",
    ),
    (
        np.random.uniform(size=(n_rounds, 7)),
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.random.uniform(size=n_rounds * len_list),
        np.ones(n_rounds * len_list * n_unique_action),  #
        "evaluation_policy_action_dist[i * n_unique_action : (i+1) * n_unique_action]",
    ),
]


valid_input_of_regression_models = [
    (
        np.random.uniform(size=(n_rounds, 7)),  #
        np.random.choice(n_unique_action, size=n_rounds * len_list),
        np.random.ones(size=n_rounds * len_list),
        np.random.ones(size=n_rounds * len_list),
        np.random.ones(size=n_rounds * len_list),
        np.ones(size=n_rounds * len_list * n_unique_action) / n_unique_action,
        "",
    ),
]


@pytest.mark.parametrize(
    "n_unique_action, len_list, fitting_method, base_model, err, description",
    invalid_input_of_initializing_regression_models,
)
def test_initializing_regression_models_using_invalid_input_data(
    n_unique_action,
    len_list,
    fitting_method,
    base_model,
    err,
    description,
) -> None:
    # initialization raises ValueError
    with pytest.raises(err, match=f"{description}*"):
        _ = SlateRegressionModel(
            n_unique_action=n_unique_action,
            len_list=len_list,
            base_model=base_model,
            fitting_method=fitting_method,
        )


@pytest.mark.parametrize(
    "context, action, reward, pscore, evaluation_policy_pscore, evaluation_policy_action_dist, err, description",
    invalid_input_of_fitting_regression_models,
)
def test_fitting_regression_models_using_invalid_input_data(
    context,
    action,
    reward,
    pscore, 
    evaluation_policy_pscore, 
    evaluation_policy_action_dist, 
    err, 
    description,
) -> None:
    # fit_predict function raises ValueError
    with pytest.raises(err, match=f"{description}*"):
        regression_model = SlateRegressionModel(
            n_unique_action=n_unique_action,
            len_list=len_list,
            base_model=Ridge(**hyperparams["ridge"]),
            fitting_method="normal",
        )
        _ = regression_model.fit_predict(
            context=context,
            action=action,
            reward=reward,
            pscore_cascade=pscore,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )
        

@pytest.mark.parametrize(
    "context, action, reward, pscore, evaluation_policy_pscore, evaluation_policy_action_dist, description",
    valid_input_of_regression_models,
)
def test_regression_models_using_valid_input_data(
    context,
    action,
    reward,
    pscore, 
    evaluation_policy_pscore, 
    evaluation_policy_action_dist, 
    description,
) -> None:
    # fit_predict
    for fitting_method in ["normal", "iw"]:
        regression_model = SlateRegressionModel(
            n_unique_action=n_unique_action,
            len_list=len_list,
            base_model=Ridge(**hyperparams["ridge"]),
            fitting_method=fitting_method,
        )
        _ = regression_model.fit_predict(
            context=context,
            action=action,
            reward=reward,
            pscore_cascade=pscore,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )

def test_slate_ope_performance_using_cascade_additive_log():
    # set parameters
    n_unique_action = 3
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 1000
    reward_structure = "cascade_additive"
    click_model = None
    behavior_policy_function = linear_behavior_policy_logit
    reward_function = logistic_reward_function
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=behavior_policy_function,
        base_reward_function=reward_function,
    )
    random_behavior_dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=None,
        base_reward_function=reward_function,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    slate_id = bandit_feedback["slate_id"]
    context = bandit_feedback["context"]
    action = bandit_feedback["action"]
    reward = bandit_feedback["reward"]
    pscore = bandit_feedback["pscore_cascade"]
    position = bandit_feedback["position"]

    # obtain random behavior feedback
    random_behavior_feedback = random_behavior_dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds
    )
    evaluation_policy_logit_ = np.ones((n_rounds, n_unique_action)) / n_unique_action
    evaluation_policy_action_dist = np.ones(n_rounds * len_list * n_unique_action) / n_unique_action
    _, _, evaluation_policy_pscore = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
        return_pscore_item_position=False,
    )
    evaluation_policy_action_dist = dataset.calc_evaluation_policy_action_dist(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
    )
    q_expected = calc_ground_truth_mean_reward_function(
        dataset=dataset,
        context=context,
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
    )

    # obtain q_hat and check if q_hat is effective
    dr_criterion_pass_rate = 0.7
    for fitting_method in ["normal", "iw"]:
        for model_name, model in model_dict.items():
            base_regression_model = SlateRegressionModel(
                base_model=model(**hyperparams[model_name]),
                len_list=len_list,
                n_unique_action=n_unique_action,
                fitting_method=fitting_method,
            )
            q_hat = base_regression_model.fit_predict(
                context=context,
                action=action,
                reward=reward,
                pscore_cascade=pscore,
                evaluation_policy_pscore_cascade=evaluation_policy_pscore,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
            )
            # compare dr criterion
            dr_criterion = np.abs((q_expected - q_hat)) - np.abs(
                q_hat
            )
            print(
                f"Dr criterion is satisfied with probability {np.mean(dr_criterion <= 0)} ------ model: {model_name} ({fitting_method}),"
            )
            assert (
                np.mean(dr_criterion <= 0) >= dr_criterion_pass_rate
            ), f" should be satisfied with a probability at least {dr_criterion_pass_rate}"


def test_slate_ope_performance_using_independent_log():
    # set parameters
    n_unique_action = 3
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 1000
    reward_structure = "independent"
    click_model = None
    behavior_policy_function = linear_behavior_policy_logit
    reward_function = logistic_reward_function
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=behavior_policy_function,
        base_reward_function=reward_function,
    )
    random_behavior_dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=None,
        base_reward_function=reward_function,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    slate_id = bandit_feedback["slate_id"]
    context = bandit_feedback["context"]
    action = bandit_feedback["action"]
    reward = bandit_feedback["reward"]
    pscore = bandit_feedback["pscore_cascade"]
    position = bandit_feedback["position"]

    # obtain random behavior feedback
    random_behavior_feedback = random_behavior_dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds
    )
    evaluation_policy_logit_ = np.ones((n_rounds, n_unique_action)) / n_unique_action
    evaluation_policy_action_dist = np.ones(n_rounds * len_list * n_unique_action) / n_unique_action
    _, _, evaluation_policy_pscore = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
        return_pscore_item_position=False,
    )
    evaluation_policy_action_dist = dataset.calc_evaluation_policy_action_dist(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
    )
    q_expected = calc_ground_truth_mean_reward_function(
        dataset=dataset,
        context=context,
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
    )

    # obtain q_hat and check if q_hat is effective
    dr_criterion_pass_rate = 0.7
    for fitting_method in ["normal", "iw"]:
        for model_name, model in model_dict.items():
            base_regression_model = SlateRegressionModel(
                base_model=model(**hyperparams[model_name]),
                len_list=len_list,
                n_unique_action=n_unique_action,
                fitting_method=fitting_method,
            )
            q_hat = base_regression_model.fit_predict(
                context=context,
                action=action,
                reward=reward,
                pscore_cascade=pscore,
                evaluation_policy_pscore_cascade=evaluation_policy_pscore,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
            )
            # compare dr criterion
            dr_criterion = np.abs((q_expected - q_hat)) - np.abs(
                q_hat
            )
            print(
                f"Dr criterion is satisfied with probability {np.mean(dr_criterion <= 0)} ------ model: {model_name} ({fitting_method}),"
            )
            assert (
                np.mean(dr_criterion <= 0) >= dr_criterion_pass_rate
            ), f" should be satisfied with a probability at least {dr_criterion_pass_rate}"


def test_slate_ope_performance_using_standard_additive_log():
    # set parameters
    n_unique_action = 3
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 1000
    reward_structure = "standard_additive"
    click_model = None
    behavior_policy_function = linear_behavior_policy_logit
    reward_function = logistic_reward_function
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=behavior_policy_function,
        base_reward_function=reward_function,
    )
    random_behavior_dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=None,
        base_reward_function=reward_function,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    slate_id = bandit_feedback["slate_id"]
    context = bandit_feedback["context"]
    action = bandit_feedback["action"]
    reward = bandit_feedback["reward"]
    pscore = bandit_feedback["pscore_cascade"]
    position = bandit_feedback["position"]

    # obtain random behavior feedback
    random_behavior_feedback = random_behavior_dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds
    )
    evaluation_policy_logit_ = np.ones((n_rounds, n_unique_action)) / n_unique_action
    evaluation_policy_action_dist = np.ones(n_rounds * len_list * n_unique_action) / n_unique_action
    _, _, evaluation_policy_pscore = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
        return_pscore_item_position=False,
    )
    evaluation_policy_action_dist = dataset.calc_evaluation_policy_action_dist(
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
    )
    q_expected = calc_ground_truth_mean_reward_function(
        dataset=dataset,
        context=context,
        action=action,
        evaluation_policy_logit_=evaluation_policy_logit_,
    )

    # obtain q_hat and check if q_hat is effective
    dr_criterion_pass_rate = 0.7
    for fitting_method in ["normal", "iw"]:
        for model_name, model in model_dict.items():
            base_regression_model = SlateRegressionModel(
                base_model=model(**hyperparams[model_name]),
                len_list=len_list,
                n_unique_action=n_unique_action,
                fitting_method=fitting_method,
            )
            q_hat = base_regression_model.fit_predict(
                context=context,
                action=action,
                reward=reward,
                pscore_cascade=pscore,
                evaluation_policy_pscore_cascade=evaluation_policy_pscore,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
            )
            # compare dr criterion
            dr_criterion = np.abs((q_expected - q_hat)) - np.abs(
                q_hat
            )
            print(
                f"Dr criterion is satisfied with probability {np.mean(dr_criterion <= 0)} ------ model: {model_name} ({fitting_method}),"
            )
            assert (
                np.mean(dr_criterion <= 0) >= dr_criterion_pass_rate
            ), f" should be satisfied with a probability at least {dr_criterion_pass_rate}"


def calc_ground_truth_mean_reward_function(
    dataset,
    context: np.ndarray,
    action: np.ndarray,
    evaluation_policy_logit_: np.ndarray,
):
    n_rounds = len(context)
    action = action.reshape((n_rounds, dataset.len_list))
    ground_truth_mean_reward_function = np.zeros_like(action, dtype=float)

    for position in range(dataset.len_list):
        if position != dataset.len_list - 1:
            if dataset.is_factorizable:
                enumerated_slate_actions = [
                    _
                    for _ in product(np.arange(dataset.n_unique_action), repeat=dataset.len_list - position - 1)
                ]
            else:
                enumerated_slate_actions = [
                    _ for _ in permutations(np.arange(dataset.n_unique_action), dataset.len_list)
                ]
            enumerated_slate_actions = np.array(enumerated_slate_actions).astype("int8")
            n_enumerated_slate_actions = len(enumerated_slate_actions)

        for i in range(n_rounds):
            if position != dataset.len_list - 1:
                action_ = np.tile(action[i][:position + 1], (n_enumerated_slate_actions, 1))
                enumerated_slate_actions = np.concatenate([action_, enumerated_slate_actions])
            else:
                enumerated_slate_actions = action[i]
                n_enumerated_slate_actions = 1

            pscores = []
            if dataset.is_factorizable:
                for action_list in enumerated_slate_actions:
                    pscores.append(
                        softmax(evaluation_policy_logit_)[:, action_list].prod(1)
                    )
                pscores = np.array(pscores).T
            else:
                pscores = np.array(
                    dataset._calc_pscore_given_policy_logit(
                        all_slate_actions=enumerated_slate_actions,
                        policy_logit_i_=evaluation_policy_logit_[i],
                    )
                )

            # calculate expected slate-level reward for each combinatorial set of items (i.e., slate actions)
            if dataset.base_reward_function is None:
                expected_slot_reward = dataset.sample_contextfree_expected_reward(
                    random_state=dataset.random_state
                )
                expected_slot_reward_tile = np.tile(
                    expected_slot_reward, (n_rounds * n_enumerated_slate_actions, 1, 1)
                )
                expected_slate_rewards = np.array(
                    [
                        expected_slot_reward_tile[
                            np.arange(n_enumerated_slate_actions) % n_enumerated_slate_actions,
                            np.array(enumerated_slate_actions)[:, position_],
                            position_,
                        ]
                        for position_ in np.arange(dataset.len_list)
                    ]
                ).T
                policy_value = (pscores * expected_slate_rewards.sum(axis=1)).sum()
            else:
                n_batch = (
                    n_rounds * n_enumerated_slate_actions * dataset.len_list - 1
                ) // 10 ** 7 + 1
                batch_size = (n_rounds - 1) // n_batch + 1
                n_batch = (n_rounds - 1) // batch_size + 1

                policy_value = 0.0
                for batch_idx in range(n_batch):
                    context_ = context[
                        batch_idx * batch_size : (batch_idx + 1) * batch_size
                    ]
                    pscores_ = pscores[
                        batch_idx * batch_size : (batch_idx + 1) * batch_size
                    ]

                    expected_slate_rewards_ = dataset.reward_function(
                        context=context_,
                        action_context=dataset.action_context,
                        action=enumerated_slate_actions.flatten(),
                        action_interaction_weight_matrix=dataset.action_interaction_weight_matrix,
                        base_reward_function=dataset.base_reward_function,
                        reward_type=dataset.reward_type,
                        reward_structure=dataset.reward_structure,
                        len_list=dataset.len_list,
                        is_enumerated=True,
                        random_state=dataset.random_state,
                    )

                    # click models based on expected reward
                    expected_slate_rewards_ *= dataset.exam_weight
                    if dataset.reward_type == "binary":
                        discount_factors = np.ones(expected_slate_rewards_.shape[0])
                        previous_slot_expected_reward = np.zeros(
                            expected_slate_rewards_.shape[0]
                        )
                        for position_ in np.arange(dataset.len_list):
                            discount_factors *= (
                                previous_slot_expected_reward
                                * dataset.attractiveness[position_]
                                + (1 - previous_slot_expected_reward)
                            )
                            expected_slate_rewards_[:, position_] = (
                                discount_factors * expected_slate_rewards_[:, position_]
                            )
                            previous_slot_expected_reward = expected_slate_rewards_[
                                :, position_
                            ]

                    policy_value += (
                        pscores_.flatten() * expected_slate_rewards_.sum(axis=1)
                    ).sum()
            ground_truth_mean_reward_function[i, position] = policy_value

    return ground_truth_mean_reward_function
