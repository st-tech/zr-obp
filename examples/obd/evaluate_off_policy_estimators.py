import argparse
import yaml
from pathlib import Path

import numpy as np
from pandas import DataFrame
from joblib import Parallel, delayed
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
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

# hyperparameters of the regression model used in model dependent OPE estimators
with open("./conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

# OPE estimators compared
ope_estimators = [DirectMethod(), InverseProbabilityWeighting(), DoublyRobust()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate off-policy estimators.")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="number of bootstrap sampling in the experiment.",
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
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="the maximum number of concurrently running jobs.",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations
    n_runs = args.n_runs
    base_model = args.base_model
    evaluation_policy = args.evaluation_policy
    behavior_policy = args.behavior_policy
    campaign = args.campaign
    n_sim_to_compute_action_dist = args.n_sim_to_compute_action_dist
    n_jobs = args.n_jobs
    random_state = args.random_state
    np.random.seed(random_state)

    obd = OpenBanditDataset(
        behavior_policy=behavior_policy,
        campaign=campaign,
    )
    # compute action distribution by evaluation policy
    kwargs = dict(
        n_actions=obd.n_actions, len_list=obd.len_list, random_state=random_state
    )
    if evaluation_policy == "bts":
        kwargs["is_zozotown_prior"] = True
        kwargs["campaign"] = campaign
    policy = evaluation_policy_dict[evaluation_policy](**kwargs)
    action_dist_single_round = policy.compute_batch_action_dist(
        n_sim=n_sim_to_compute_action_dist
    )
    # ground-truth policy value of an evaluation policy
    # , which is estimated with factual (observed) rewards (on-policy estimation)
    ground_truth_policy_value = OpenBanditDataset.calc_on_policy_policy_value_estimate(
        behavior_policy=evaluation_policy,
        campaign=campaign,
    )

    def process(b: int):
        # sample bootstrap from batch logged bandit feedback
        bandit_feedback = obd.sample_bootstrap_bandit_feedback(random_state=b)
        # estimate the mean reward function with an ML model
        regression_model = RegressionModel(
            n_actions=obd.n_actions,
            len_list=obd.len_list,
            action_context=obd.action_context,
            base_model=base_model_dict[base_model](**hyperparams[base_model]),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=bandit_feedback["context"],
            action=bandit_feedback["action"],
            reward=bandit_feedback["reward"],
            position=bandit_feedback["position"],
            pscore=bandit_feedback["pscore"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )
        # evaluate estimators' performances using relative estimation error (relative-ee)
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
    relative_ee_dict = {est.estimator_name: dict() for est in ope_estimators}
    for b, relative_ee_b in enumerate(processed):
        for (
            estimator_name,
            relative_ee_,
        ) in relative_ee_b.items():
            relative_ee_dict[estimator_name][b] = relative_ee_
    relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)

    print("=" * 30)
    print(f"random_state={random_state}")
    print("-" * 30)
    print(relative_ee_df[["mean", "std"]])
    print("=" * 30)

    # save results of the evaluation of off-policy estimators in './logs' directory.
    log_path = Path("./logs") / behavior_policy / campaign
    log_path.mkdir(exist_ok=True, parents=True)
    relative_ee_df.to_csv(log_path / "relative_ee_of_ope_estimators.csv")
