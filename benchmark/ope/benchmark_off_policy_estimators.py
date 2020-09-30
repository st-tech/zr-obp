import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS, Random
from obp.ope import (
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    SwitchDoublyRobust,
    DirectMethod,
    DoublyRobust,
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
        "--base_model",
        type=str,
        choices=["logistic_regression", "lightgbm"],
        required=True,
        help="base ML model for regression model, logistic_regression or lightgbm.",
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
        "--n_sim_for_action_dist",
        type=float,
        default=1000000,
        help="number of monte carlo simulation to compute the action distribution of bts.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="the proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--is_timeseries_split",
        action="store_true",
        help="If true, split the original logged badnit feedback data by time series.",
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

    # configurations of the benchmark experiment
    n_boot_samples = args.n_boot_samples
    base_model = args.base_model
    behavior_policy = args.behavior_policy
    evaluation_policy = "bts" if behavior_policy == "random" else "random"
    campaign = args.campaign
    n_sim_for_action_dist = args.n_sim_for_action_dist
    test_size = args.test_size
    is_timeseries_split = args.is_timeseries_split
    n_sim_to_compute_action_dist = args.n_sim_to_compute_action_dist
    random_state = args.random_state
    data_path = Path("../open_bandit_dataset")
    # prepare path
    log_path = (
        Path("./logs") / behavior_policy / campaign / "out_sample" / base_model
        if is_timeseries_split
        else Path("./logs") / behavior_policy / campaign / "in_sample" / base_model
    )
    reg_model_path = log_path / "trained_reg_models"
    reg_model_path.mkdir(exist_ok=True, parents=True)

    obd = OpenBanditDataset(
        behavior_policy=behavior_policy, campaign=campaign, data_path=data_path
    )
    # compared OPE estimators
    ope_estimators = [
        DirectMethod(),
        InverseProbabilityWeighting(),
        SelfNormalizedInverseProbabilityWeighting(),
        DoublyRobust(),
        SwitchDoublyRobust(tau=0.1, estimator_name="switch-dr(0.1)"),
        SwitchDoublyRobust(tau=1.0, estimator_name="switch-dr(1.0)"),
        SwitchDoublyRobust(tau=10, estimator_name="switch-dr(10)"),
        SwitchDoublyRobust(tau=100, estimator_name="switch-dr(100)"),
    ]
    # ground-truth policy value of a evaluation policy
    # , which is estimated with factual (observed) rewards (on-policy estimation)
    ground_truth_policy_value = OpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy=evaluation_policy,
        campaign=campaign,
        data_path=data_path,
        test_size=test_size,
        is_timeseries_split=is_timeseries_split,
    )

    start = time.time()
    relative_ee = {
        est.estimator_name: np.zeros(n_boot_samples) for est in ope_estimators
    }
    for b in np.arange(n_boot_samples):
        # load the pre-trained regression model
        with open(reg_model_path / f"reg_model_{b}.pkl", "rb") as f:
            reg_model = pickle.load(f)
        with open(reg_model_path / f"is_for_reg_model_{b}.pkl", "rb") as f:
            is_for_reg_model = pickle.load(f)
        # sample bootstrap from batch logged bandit feedback
        boot_bandit_feedback = obd.sample_bootstrap_bandit_feedback(
            test_size=test_size, is_timeseries_split=is_timeseries_split, random_state=b
        )
        for key_ in ["context", "action", "reward", "pscore", "position"]:
            boot_bandit_feedback[key_] = boot_bandit_feedback[key_][~is_for_reg_model]
        if evaluation_policy == "bts":
            policy = BernoulliTS(
                n_actions=obd.n_actions,
                len_list=obd.len_list,
                is_zozotown_prior=True,  # replicate the policy in the ZOZOTOWN production
                campaign=campaign,
                random_state=random_state,
            )
            action_dist = policy.compute_batch_action_dist(
                n_sim=100000, n_rounds=boot_bandit_feedback["n_rounds"]
            )
        else:
            policy = Random(
                n_actions=obd.n_actions,
                len_list=obd.len_list,
                random_state=random_state,
            )
            action_dist = policy.compute_batch_action_dist(
                n_sim=100000, n_rounds=boot_bandit_feedback["n_rounds"]
            )
        # evaluate the estimation performance of OPE estimators
        ope = OffPolicyEvaluation(
            bandit_feedback=boot_bandit_feedback,
            regression_model=reg_model,
            ope_estimators=ope_estimators,
        )
        estimated_policy_values = ope.estimate_policy_values(action_dist=action_dist,)
        relative_estimation_errors = ope.evaluate_performance_of_estimators(
            action_dist=action_dist,
            ground_truth_policy_value=ground_truth_policy_value,
        )
        # store relative estimation errors of OPE estimators at each bootstrap
        for (
            estimator_name,
            relative_estimation_error,
        ) in relative_estimation_errors.items():
            relative_ee[estimator_name][b] = relative_estimation_error

        print(f"{b+1}th iteration: {np.round((time.time() - start) / 60, 2)}min")

    # estimate confidence intervals of relative estimation by nonparametric bootstrap method
    evaluation_of_ope_results = {est.estimator_name: dict() for est in ope_estimators}
    for estimator_name in evaluation_of_ope_results.keys():
        evaluation_of_ope_results[estimator_name]["mean"] = relative_ee[
            estimator_name
        ].mean()
        evaluation_of_ope_results[estimator_name]["std"] = np.std(
            relative_ee[estimator_name], ddof=1
        )

    evaluation_of_ope_results_df = pd.DataFrame(evaluation_of_ope_results).T
    print("=" * 50)
    print(f"random_state={random_state}")
    print("-" * 50)
    print(evaluation_of_ope_results_df)
    print("=" * 50)

    # save results of the evaluation of off-policy estimators in './logs' directory.
    evaluation_of_ope_results_df.to_csv(log_path / f"relative_ee_of_ope_estimators.csv")

