import argparse
import yaml
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from obp.dataset import OpenBanditDataset
from obp.policy import Random, BernoulliTS
from obp.ope import (
    RegressionModel,
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
)

evaluation_policy_dict = dict(bts=BernoulliTS, random=Random)

# hyperparameter for the regression model used in model dependent OPE estimators
with open("./conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate off-policy estimators.")
    parser.add_argument(
        "--n_boot_samples",
        type=int,
        default=1,
        help="number of bootstrap samples in the experiment.",
    )
    parser.add_argument(
        "--evaluation_policy",
        type=str,
        choices=["bts", "random"],
        required=True,
        help="evaluation policy, bts or random.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for regression model, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--behavior_policy",
        type=str,
        choices=["bts", "random"],
        required=True,
        help="behavior policy, bts or random.",
    )
    parser.add_argument(
        "--campaign",
        type=str,
        choices=["all", "men", "women"],
        required=True,
        help="campaign name, men, women, or all.",
    )
    parser.add_argument(
        "--n_sim_to_compute_action_dist",
        type=float,
        default=1000000,
        help="number of monte carlo simulation to compute the action distribution of bts.",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    n_boot_samples = args.n_boot_samples
    base_model = args.base_model
    evaluation_policy = args.evaluation_policy
    behavior_policy = args.behavior_policy
    campaign = args.campaign
    n_sim_to_compute_action_dist = args.n_sim_to_compute_action_dist
    random_state = args.random_state
    np.random.seed(random_state)
    data_path = Path(".").resolve().parents[1] / "obd"

    obd = OpenBanditDataset(
        behavior_policy=behavior_policy, campaign=campaign, data_path=data_path
    )

    # hyparparameters for evaluation policies
    kwargs = dict(
        n_actions=obd.n_actions, len_list=obd.len_list, random_state=random_state
    )
    if evaluation_policy == "bts":
        kwargs["is_zozotown_prior"] = True
        kwargs["campaign"] = campaign
    policy = evaluation_policy_dict[evaluation_policy](**kwargs)
    # compared OPE estimators
    ope_estimators = [DirectMethod(), InverseProbabilityWeighting(), DoublyRobust()]
    # ground-truth policy value of an evaluation policy
    # , which is estimated with factual (observed) rewards (on-policy estimation)
    ground_truth_policy_value = OpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy=evaluation_policy, campaign=campaign, data_path=data_path
    )

    relative_ee = {
        est.estimator_name: np.zeros(n_boot_samples) for est in ope_estimators
    }
    start = time.time()
    for b in np.arange(n_boot_samples):
        # sample bootstrap from batch logged bandit feedback
        boot_bandit_feedback = obd.sample_bootstrap_bandit_feedback(random_state=b)
        if evaluation_policy == "bts":
            action_dist = policy.compute_batch_action_dist(
                n_sim=n_sim_to_compute_action_dist,
                n_rounds=boot_bandit_feedback["n_rounds"],
            )
        else:
            action_dist = policy.compute_batch_action_dist(
                n_rounds=boot_bandit_feedback["n_rounds"],
            )
        # evaluate the estimation performance of OPE estimators by relative estimation errors
        ope = OffPolicyEvaluation(
            bandit_feedback=boot_bandit_feedback,
            regression_model=RegressionModel(
                n_actions=obd.n_actions,
                len_list=obd.len_list,
                action_context=obd.action_context,
                base_model=base_model_dict[base_model](**hyperparams[base_model]),
            ),
            ope_estimators=ope_estimators,
        )
        relative_estimation_errors = ope.evaluate_performance_of_estimators(
            action_dist=action_dist,
            ground_truth_policy_value=ground_truth_policy_value,
        )
        policy.initialize()
        # store relative estimation errors of OPE estimators estimated with each sample
        for (
            estimator_name,
            relative_estimation_error,
        ) in relative_estimation_errors.items():
            relative_ee[estimator_name][b] = relative_estimation_error

        print(f"{b+1}th iteration: {np.round((time.time() - start) / 60, 2)}min")

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
    print("=" * 30)
    print(f"random_state={random_state}")
    print("-" * 30)
    print(evaluation_of_ope_results_df)
    print("=" * 30)

    # save results of the evaluation of off-policy estimators in './logs' directory.
    log_path = Path("./logs") / behavior_policy / campaign
    log_path.mkdir(exist_ok=True, parents=True)
    evaluation_of_ope_results_df.to_csv(log_path / "relative_ee_of_ope_estimators.csv")
