import argparse
import time
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from obp.dataset import (
    SyntheticBanditDataset,
    linear_behavior_policy,
    logistic_reward_function,
)
from obp.policy import IPWLearner
from obp.ope import (
    RegressionModel,
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
    SelfNormalizedDoublyRobust,
    SwitchDoublyRobust,
    SwitchInverseProbabilityWeighting,
    DoublyRobustWithShrinkage,
)


# hyperparameter for the regression model used in model dependent OPE estimators
with open("./conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate off-policy estimators with synthetic data."
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="number of simulations in the experiment."
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=10000,
        help="number of rounds for synthetic bandit feedback.",
    )
    parser.add_argument(
        "--n_actions",
        type=int,
        default=10,
        help="number of actions for synthetic bandit feedback.",
    )
    parser.add_argument(
        "--dim_context",
        type=int,
        default=5,
        help="dimensions of context vectors characterizing each round.",
    )
    parser.add_argument(
        "--dim_action_context",
        type=int,
        default=5,
        help="dimensions of context vectors characterizing each action.",
    )
    parser.add_argument(
        "--base_model_for_evaluation_policy",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for evaluation policy, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--base_model_for_reg_model",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for regression model, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations
    n_runs = args.n_runs
    n_rounds = args.n_rounds
    n_actions = args.n_actions
    dim_context = args.dim_context
    dim_action_context = args.dim_action_context
    base_model_for_evaluation_policy = args.base_model_for_evaluation_policy
    base_model_for_reg_model = args.base_model_for_reg_model
    random_state = args.random_state
    np.random.seed(random_state)

    # synthetic data generator
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        dim_action_context=dim_action_context,
        reward_function=logistic_reward_function,
        behavior_policy_function=linear_behavior_policy,
        random_state=random_state,
    )
    # define evaluation policy using IPWLearner
    evaluation_policy = IPWLearner(
        n_actions=dataset.n_actions,
        len_list=dataset.len_list,
        base_model=base_model_dict[base_model_for_evaluation_policy](
            **hyperparams[base_model_for_evaluation_policy]
        ),
    )
    # compared OPE estimators
    ope_estimators = [
        DirectMethod(),
        InverseProbabilityWeighting(),
        SelfNormalizedInverseProbabilityWeighting(),
        DoublyRobust(),
        SelfNormalizedDoublyRobust(),
        SwitchInverseProbabilityWeighting(tau=1, estimator_name="switch-ipw (tau=1)"),
        SwitchInverseProbabilityWeighting(
            tau=100, estimator_name="switch-ipw (tau=100)"
        ),
        SwitchDoublyRobust(tau=1, estimator_name="switch-dr (tau=1)"),
        SwitchDoublyRobust(tau=100, estimator_name="switch-dr (tau=100)"),
        DoublyRobustWithShrinkage(lambda_=1, estimator_name="dr-os (lambda=1)"),
        DoublyRobustWithShrinkage(lambda_=100, estimator_name="dr-os (lambda=100)"),
    ]

    start = time.time()
    relative_ee = {est.estimator_name: np.zeros(n_runs) for est in ope_estimators}
    for i in np.arange(n_runs):
        # sample new training and test sets of synthetic logged bandit feedback
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        # train the evaluation policy on the training set of the synthetic logged bandit feedback
        evaluation_policy.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
        )
        # predict the action decisions for the test set of the synthetic logged bandit feedback
        action_dist = evaluation_policy.predict_proba(
            context=bandit_feedback_test["context"]
        )
        # estimate the ground-truth policy values of the evaluation policy
        # using the full expected reward contained in the bandit feedback dictionary
        ground_truth_policy_value = np.average(
            bandit_feedback_test["expected_reward"],
            weights=action_dist[:, :, 0],
            axis=1,
        ).mean()
        # estimate the mean reward function with an ML model
        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            len_list=dataset.len_list,
            action_context=dataset.action_context,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            position=bandit_feedback_train["position"],
            pscore=bandit_feedback_train["pscore"],
            n_folds=3,  # 3-fold cross-fitting
        )
        # evaluate the estimation performance of OPE estimators
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback_test, ope_estimators=ope_estimators,
        )
        relative_estimation_errors = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        # store relative estimation errors of OPE estimators at each split
        for (
            estimator_name,
            relative_estimation_error,
        ) in relative_estimation_errors.items():
            relative_ee[estimator_name][i] = relative_estimation_error

        print(f"{i+1}th iteration: {np.round((time.time() - start) / 60, 2)}min")

    # estimate mean and standard deviations of the relative estimation errors
    evaluation_of_ope_results = {est.estimator_name: dict() for est in ope_estimators}
    for estimator_name in evaluation_of_ope_results.keys():
        evaluation_of_ope_results[estimator_name]["mean"] = relative_ee[
            estimator_name
        ].mean()
        evaluation_of_ope_results[estimator_name]["std"] = np.std(
            relative_ee[estimator_name], ddof=1
        )
    evaluation_of_ope_results_df = pd.DataFrame(evaluation_of_ope_results).T
    print("=" * 40)
    print(f"random_state={random_state}")
    print("-" * 40)
    print(evaluation_of_ope_results_df)
    print("=" * 40)

    # save results of the evaluation of off-policy estimators in './logs' directory.
    log_path = Path("./logs")
    log_path.mkdir(exist_ok=True, parents=True)
    evaluation_of_ope_results_df.to_csv(log_path / "relative_ee_of_ope_estimators.csv")
