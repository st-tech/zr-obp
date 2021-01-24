from typing import Dict
from pathlib import Path
import yaml

import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from obp.ope import RegressionModel
from obp.types import BanditFeedback


binary_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

# hyperparameter settings for the base ML model in regression model
cd_path = Path(__file__).parent.resolve()
with open(cd_path / "hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)


def test_performance_of_binary_outcome_models(
    fixed_synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the performance of ope estimators using synthetic bandit data and random evaluation policy
    when the regression model is estimated by a logistic regression
    """
    bandit_feedback = fixed_synthetic_bandit_feedback.copy()
    expected_reward = np.expand_dims(bandit_feedback["expected_reward"], axis=-1)
    action_dist = random_action_dist
    # compute ground truth policy value using expected reward
    ground_truth_policy_value = np.average(
        expected_reward[:, :, 0], weights=action_dist[:, :, 0], axis=1
    )
    # compute statistics of ground truth policy value
    gt_mean = ground_truth_policy_value.mean()
    gt_std = ground_truth_policy_value.std(ddof=1)
    random_state = 12345
    auc_scores: Dict[str, float] = {}
    # check ground truth
    ci_times = 5
    ci_bound = gt_std * ci_times / np.sqrt(ground_truth_policy_value.shape[0])
    print(
        f"gt_mean: {gt_mean}, {ci_times} * gt_std / sqrt({ground_truth_policy_value.shape[0]}): {ci_bound}"
    )
    # check the performance of regression models using doubly robust criteria (|\hat{q} - q| <= |q| is satisfied with high probability)
    dr_criteria_pass_rate = 0.8
    fit_methods = ["normal", "iw", "mrdr"]
    for fit_method in fit_methods:
        for model_name, model in binary_model_dict.items():
            regression_model = RegressionModel(
                n_actions=bandit_feedback["n_actions"],
                len_list=bandit_feedback["position"].ndim,
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
                    position=bandit_feedback["position"],
                    action_dist=action_dist,
                    n_folds=3,  # 3-fold cross-fitting
                    random_state=random_state,
                )
            auc_scores[model_name + "_" + fit_method] = roc_auc_score(
                y_true=bandit_feedback["reward"],
                y_score=estimated_rewards_by_reg_model[
                    np.arange(bandit_feedback["reward"].shape[0]),
                    bandit_feedback["action"],
                    bandit_feedback["position"],
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
            ), f"Dr criteria should not be larger then 0 with probability {dr_criteria_pass_rate}"

    for model_name in auc_scores:
        print(f"AUC of {model_name} is {auc_scores[model_name]}")
        assert (
            auc_scores[model_name] > 0.5
        ), f"AUC of {model_name} should be greator than 0.5"
