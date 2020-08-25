import time
import argparse
from pathlib import Path
import yaml
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from obp.dataset import OpenBanditDataset
from obp.ope import RegressionModel
from obp.utils import estimate_confidence_interval_by_bootstrap


with open("./conf/hyperparam.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression, lightgbm=HistGradientBoostingClassifier
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
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations of the benchmark experiment
    n_boot_samples = args.n_boot_samples
    base_model = args.base_model
    behavior_policy = args.behavior_policy
    campaign = args.campaign
    random_state = args.random_state
    data_path = Path("../open_bandit_dataset")
    # prepare path
    log_path = Path("./logs") / behavior_policy / campaign / base_model
    reg_model_path = log_path / "trained_reg_models"
    reg_model_path.mkdir(exist_ok=True, parents=True)

    obd = OpenBanditDataset(
        behavior_policy=behavior_policy, campaign=campaign, data_path=data_path
    )
    # a base ML model for regression model
    reg_model = RegressionModel(
        base_model=CalibratedClassifierCV(
            base_model_dict[base_model](**hyperparams[base_model])
        )
    )

    start_time = time.time()
    performance_of_reg_model = {
        metrics[i]: np.zeros(n_boot_samples) for i in np.arange(len(metrics))
    }
    for b in np.arange(n_boot_samples):
        # sample bootstrap from batch logged bandit feedback
        boot_bandit_feedback = obd.sample_bootstrap_bandit_feedback(random_state=b)
        # train a regression model on logged bandit feedback data
        reg_model.fit(
            context=boot_bandit_feedback["context"],
            action=boot_bandit_feedback["action"],
            reward=boot_bandit_feedback["reward"],
            pscore=boot_bandit_feedback["pscore"],
            action_context=boot_bandit_feedback["action_context"],
        )
        # evaluate the (in-sample) estimation performance of the regression model by AUC and RCE
        predicted_rewards = reg_model.predict(
            context=boot_bandit_feedback["context"],
            action_context=boot_bandit_feedback["action_context"],
            selected_actions=np.expand_dims(boot_bandit_feedback["action"], 1),
            position=np.zeros(boot_bandit_feedback["n_rounds"], dtype=int),
        )
        rewards = boot_bandit_feedback["reward"]
        performance_of_reg_model["auc"][b] = roc_auc_score(
            y_true=rewards, y_score=predicted_rewards
        )
        rce_mean = log_loss(
            y_true=rewards, y_pred=np.ones_like(rewards) * np.mean(rewards)
        )
        rce_clf = log_loss(y_true=rewards, y_pred=predicted_rewards)
        performance_of_reg_model["rce"][b] = (rce_mean - rce_clf) / rce_clf

        # save trained regression model in a pickled form
        pickle.dump(
            reg_model, open(reg_model_path / f"reg_model_{b}.pkl", "wb"),
        )
        print(
            f"Finished {b+1}th bootstrap sample:",
            f"{np.round((time.time() - start_time) / 60, 1)}min",
        )

    # estimate confidence intervals of the performances of the regression model
    performance_of_reg_model_with_ci = {}
    for metric in metrics:
        performance_of_reg_model_with_ci[
            metric
        ] = estimate_confidence_interval_by_bootstrap(
            samples=performance_of_reg_model[metric], random_state=random_state
        )
    performance_of_reg_model_df = pd.DataFrame(performance_of_reg_model_with_ci).T

    print("=" * 50)
    print(f"random_state={random_state}")
    print("-" * 50)
    print(performance_of_reg_model_df)
    print("=" * 50)

    # save performance of the regression model in './logs' directory.
    performance_of_reg_model_df.to_csv(log_path / f"performance_of_reg_model.csv")
