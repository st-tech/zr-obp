import time
import argparse
import yaml
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from obp.dataset import OpenBanditDataset
from obp.ope import RegressionModel

# hyperparameter settings for the base ML model in regression model
with open("./conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression, lightgbm=HistGradientBoostingClassifier,
)

metrics = ["auc", "rce"]

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
        help="base ML model for regression model, logistic_regression, or lightgbm.",
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
        action="store_true",
        help="If true, split the original logged badnit feedback data by time series.",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations of the benchmark experiment
    n_boot_samples = args.n_boot_samples
    base_model = args.base_model
    behavior_policy = args.behavior_policy
    campaign = args.campaign
    test_size = args.test_size
    is_timeseries_split = args.is_timeseries_split
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
    start_time = time.time()
    performance_of_reg_model = {
        metrics[i]: np.zeros(n_boot_samples) for i in np.arange(len(metrics))
    }
    for b in np.arange(n_boot_samples):
        # sample bootstrap samples from batch logged bandit feedback
        boot_bandit_feedback = obd.sample_bootstrap_bandit_feedback(
            test_size=test_size, is_timeseries_split=is_timeseries_split, random_state=b
        )
        # split data into two folds (data for training reg_model and for ope)
        is_for_reg_model = np.random.binomial(
            n=1, p=0.3, size=boot_bandit_feedback["n_rounds"]
        ).astype(bool)
        # define regression model
        reg_model = RegressionModel(
            n_actions=obd.n_actions,
            len_list=obd.len_list,
            action_context=boot_bandit_feedback["action_context"],
            base_model=base_model_dict[base_model](**hyperparams[base_model]),
        )
        # train regression model on logged bandit feedback data
        reg_model.fit(
            context=boot_bandit_feedback["context"][is_for_reg_model],
            action=boot_bandit_feedback["action"][is_for_reg_model],
            reward=boot_bandit_feedback["reward"][is_for_reg_model],
            position=boot_bandit_feedback["position"][is_for_reg_model],
        )
        # evaluate the estimation performance of the regression model by AUC and RCE
        if is_timeseries_split:
            estimated_reward_by_reg_model = reg_model.predict(
                context=boot_bandit_feedback["context_test"],
            )
            rewards = boot_bandit_feedback["reward_test"]
            estimated_rewards_ = estimated_reward_by_reg_model[
                np.arange(rewards.shape[0]),
                boot_bandit_feedback["action_test"].astype(int),
                boot_bandit_feedback["position_test"].astype(int),
            ]
        else:
            estimated_reward_by_reg_model = reg_model.predict(
                context=boot_bandit_feedback["context"][~is_for_reg_model],
            )
            rewards = boot_bandit_feedback["reward"][~is_for_reg_model]
            estimated_rewards_ = estimated_reward_by_reg_model[
                np.arange((~is_for_reg_model).sum()),
                boot_bandit_feedback["action"][~is_for_reg_model].astype(int),
                boot_bandit_feedback["position"][~is_for_reg_model].astype(int),
            ]
        performance_of_reg_model["auc"][b] = roc_auc_score(
            y_true=rewards, y_score=estimated_rewards_
        )
        rce_naive = -log_loss(
            y_true=rewards,
            y_pred=np.ones_like(rewards)
            * boot_bandit_feedback["reward"][is_for_reg_model].mean(),
        )
        rce_clf = -log_loss(y_true=rewards, y_pred=estimated_rewards_)
        performance_of_reg_model["rce"][b] = (rce_naive - rce_clf) / rce_naive

        # save trained regression model in a pickled form
        pickle.dump(
            reg_model, open(reg_model_path / f"reg_model_{b}.pkl", "wb"),
        )
        pickle.dump(
            is_for_reg_model, open(reg_model_path / f"is_for_reg_model_{b}.pkl", "wb"),
        )
        print(
            f"Finished {b+1}th bootstrap sample:",
            f"{np.round((time.time() - start_time) / 60, 1)}min",
        )

    # estimate means and standard deviations of the performances of the regression model
    performance_of_reg_model_ = {metric: dict() for metric in metrics}
    for metric in performance_of_reg_model_.keys():
        performance_of_reg_model_[metric]["mean"] = performance_of_reg_model[
            metric
        ].mean()
        performance_of_reg_model_[metric]["std"] = np.std(
            performance_of_reg_model[metric], ddof=1
        )
    performance_of_reg_model_df = pd.DataFrame(performance_of_reg_model_).T

    print("=" * 50)
    print(f"random_state={random_state}")
    print("-" * 50)
    print(performance_of_reg_model_df)
    print("=" * 50)

    # save performance of the regression model in './logs' directory.
    performance_of_reg_model_df.to_csv(log_path / f"performance_of_reg_model.csv")
