import argparse
import pickle
from pathlib import Path
from distutils.util import strtobool

import numpy as np
from pandas import DataFrame
from joblib import Parallel, delayed

from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS, Random
from obp.ope import (
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
    SelfNormalizedDoublyRobust,
    SwitchDoublyRobust,
    DoublyRobustWithShrinkage,
)

# OPE estimators compared
ope_estimators = [
    DirectMethod(),
    InverseProbabilityWeighting(),
    SelfNormalizedInverseProbabilityWeighting(),
    DoublyRobust(),
    DoublyRobust(estimator_name="mrdr"),
    SelfNormalizedDoublyRobust(),
    SwitchDoublyRobust(tau=5.0, estimator_name="switch-dr (tau=5)"),
    SwitchDoublyRobust(tau=10.0, estimator_name="switch-dr (tau=10)"),
    SwitchDoublyRobust(tau=50.0, estimator_name="switch-dr (tau=50)"),
    SwitchDoublyRobust(tau=100.0, estimator_name="switch-dr (tau=100)"),
    SwitchDoublyRobust(tau=500.0, estimator_name="switch-dr (tau=500)"),
    SwitchDoublyRobust(tau=1000.0, estimator_name="switch-dr (tau=1000)"),
    DoublyRobustWithShrinkage(lambda_=5.0, estimator_name="dr-os (lambda=5)"),
    DoublyRobustWithShrinkage(lambda_=10.0, estimator_name="dr-os (lambda=10)"),
    DoublyRobustWithShrinkage(lambda_=50.0, estimator_name="dr-os (lambda=50)"),
    DoublyRobustWithShrinkage(lambda_=100.0, estimator_name="dr-os (lambda=100)"),
    DoublyRobustWithShrinkage(lambda_=500.0, estimator_name="dr-os (lambda=500)"),
    DoublyRobustWithShrinkage(lambda_=1000.0, estimator_name="dr-os (lambda=1000)"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate off-policy estimators.")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="number of experimental runs.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        choices=["logistic_regression", "random_forest", "lightgbm"],
        required=True,
        help="base ML model for regression model, logistic_regression, random_forest, or lightgbm.",
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
        "--test_size",
        type=float,
        default=0.3,
        help="the proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--is_timeseries_split",
        type=strtobool,
        default=False,
        help="If true, split the original logged badnit feedback data by time series.",
    )
    parser.add_argument(
        "--n_sim_to_compute_action_dist",
        type=float,
        default=1000000,
        help="number of monte carlo simulation to compute the action distribution of bts.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="the maximum number of concurrently running jobs.",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations of the benchmark experiment
    n_runs = args.n_runs
    base_model = args.base_model
    behavior_policy = args.behavior_policy
    evaluation_policy = "bts" if behavior_policy == "random" else "random"
    campaign = args.campaign
    test_size = args.test_size
    is_timeseries_split = args.is_timeseries_split
    n_sim_to_compute_action_dist = args.n_sim_to_compute_action_dist
    n_jobs = args.n_jobs
    random_state = args.random_state
    np.random.seed(random_state)
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
    # ground-truth policy value of a evaluation policy
    # , which is estimated with factual (observed) rewards (on-policy estimation)
    ground_truth_policy_value = OpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy=evaluation_policy,
        campaign=campaign,
        data_path=data_path,
        test_size=test_size,
        is_timeseries_split=is_timeseries_split,
    )
    # compute action distribution by evaluation policy
    if evaluation_policy == "bts":
        policy = BernoulliTS(
            n_actions=obd.n_actions,
            len_list=obd.len_list,
            is_zozotown_prior=True,  # replicate the policy in the ZOZOTOWN production
            campaign=campaign,
            random_state=random_state,
        )
    else:
        policy = Random(
            n_actions=obd.n_actions,
            len_list=obd.len_list,
            random_state=random_state,
        )
    action_dist_single_round = policy.compute_batch_action_dist(
        n_sim=n_sim_to_compute_action_dist
    )

    def process(b: int):
        # load the pre-trained regression model
        with open(reg_model_path / f"reg_model_{b}.pkl", "rb") as f:
            reg_model = pickle.load(f)
        with open(reg_model_path / f"reg_model_mrdr_{b}.pkl", "rb") as f:
            reg_model_mrdr = pickle.load(f)
        with open(reg_model_path / f"is_for_reg_model_{b}.pkl", "rb") as f:
            is_for_reg_model = pickle.load(f)
        # sample bootstrap samples from batch logged bandit feedback
        if is_timeseries_split:
            bandit_feedback = obd.sample_bootstrap_bandit_feedback(
                test_size=test_size,
                is_timeseries_split=is_timeseries_split,
                random_state=b,
            )
        else:
            bandit_feedback = obd.sample_bootstrap_bandit_feedback(
                test_size=test_size,
                is_timeseries_split=is_timeseries_split,
                random_state=b,
            )
            bandit_feedback["n_rounds"] = (~is_for_reg_model).sum()
            for key_ in ["context", "action", "reward", "pscore", "position"]:
                bandit_feedback[key_] = bandit_feedback[key_][~is_for_reg_model]
        # estimate the mean reward function using the pre-trained reg_model
        estimated_rewards_by_reg_model_default = reg_model.predict(
            context=bandit_feedback["context"],
        )
        estimated_rewards_by_reg_model_mrdr = reg_model_mrdr.predict(
            context=bandit_feedback["context"],
        )
        estimated_rewards_by_reg_model = {
            estimator.estimator_name: estimated_rewards_by_reg_model_mrdr
            if estimator.estimator_name == "mrdr"
            else estimated_rewards_by_reg_model_default
            for estimator in ope_estimators
        }
        # evaluate the estimation performance of OPE estimators
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback,
            ope_estimators=ope_estimators,
        )
        action_dist = np.tile(
            action_dist_single_round, (bandit_feedback["n_rounds"], 1, 1)
        )
        relative_ee_b = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        return relative_ee_b

    processed = Parallel(
        n_jobs=n_jobs,
        verbose=50,
    )([delayed(process)(i) for i in np.arange(n_runs)])

    # save results of the evaluation of ope in './logs' directory.
    relative_ee = {est.estimator_name: np.zeros(n_runs) for est in ope_estimators}
    for b, relative_ee_b in enumerate(processed):
        for (
            estimator_name,
            relative_ee_,
        ) in relative_ee_b.items():
            relative_ee[estimator_name][b] = relative_ee_
    DataFrame(relative_ee).describe().T.round(6).to_csv(
        log_path / "eval_ope_results.csv"
    )
