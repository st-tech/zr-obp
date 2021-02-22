import argparse
from copy import deepcopy
import yaml
import pickle
from distutils.util import strtobool
from pathlib import Path
from typing import Dict

import numpy as np
from pandas import DataFrame
from joblib import Parallel, delayed
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS, Random
from obp.ope import RegressionModel
from obp.types import BanditFeedback

# hyperparameter settings for the base ML model in regression model
with open("./conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)


def relative_ce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate relative cross-entropy."""
    naive_pred = np.ones_like(y_true) * y_true.mean()
    ce_naive_pred = log_loss(y_true=y_true, y_pred=naive_pred)
    ce_y_pred = log_loss(y_true=y_true, y_pred=y_pred)
    return 1.0 - (ce_y_pred / ce_naive_pred)


def evaluate_reg_model(
    bandit_feedback: BanditFeedback,
    estimated_rewards_by_reg_model: np.ndarray,
) -> Dict[str, float]:
    """Evaluate the estimation performance of regression model by AUC and RCE."""
    performance_reg_model = dict(auc=0.0, rce=0.0)
    factual_rewards = bandit_feedback["reward"]
    estimated_factual_rewards = estimated_rewards_by_reg_model[
        np.arange(factual_rewards.shape[0]),
        bandit_feedback["action"].astype(int),
        bandit_feedback["position"].astype(int),
    ]
    performance_reg_model["auc"] = roc_auc_score(
        y_true=factual_rewards, y_score=estimated_factual_rewards
    )
    performance_reg_model["rce"] = relative_ce(
        y_true=factual_rewards, y_pred=estimated_factual_rewards
    )
    return performance_reg_model


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
        choices=["logistic_regression", "lightgbm", "random_forest"],
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
        "--is_mrdr",
        type=strtobool,
        default=False,
        help="If true, the regression model is trained by minimizing the empirical variance objective.",
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
    campaign = args.campaign
    test_size = args.test_size
    is_timeseries_split = args.is_timeseries_split
    is_mrdr = args.is_mrdr
    fitting_method = "mrdr" if is_mrdr else "normal"
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
    # action distribution by evaluation policy
    # (more robust doubly robust needs evaluation policy information)
    if behavior_policy == "random":
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

    def process(b: int) -> Dict[str, float]:
        # sample bootstrap from batch logged bandit feedback
        if is_timeseries_split:
            bandit_feedback_train = obd.sample_bootstrap_bandit_feedback(
                test_size=test_size,
                is_timeseries_split=True,
                random_state=b,
            )
            bandit_feedback_test = obd.obtain_batch_bandit_feedback(
                test_size=test_size,
                is_timeseries_split=True,
            )[1]
        else:
            bandit_feedback_train = obd.sample_bootstrap_bandit_feedback(
                random_state=b,
            )
            bandit_feedback_test = deepcopy(bandit_feedback_train)
            # split data into two folds (data for training reg_model and for ope)
            is_for_reg_model = np.random.binomial(
                n=1, p=0.3, size=bandit_feedback_train["n_rounds"]
            ).astype(bool)
            with open(reg_model_path / f"is_for_reg_model_{b}.pkl", "wb") as f:
                pickle.dump(
                    is_for_reg_model,
                    f,
                )
            bandit_feedback_train["n_rounds"] = is_for_reg_model.sum()
            bandit_feedback_test["n_rounds"] = (~is_for_reg_model).sum()
            for key in ["context", "action", "reward", "pscore", "position"]:
                bandit_feedback_train[key] = bandit_feedback_train[key][
                    is_for_reg_model
                ]
                bandit_feedback_test[key] = bandit_feedback_test[key][~is_for_reg_model]
        model_file_name = f"reg_model_mrdr_{b}.pkl" if is_mrdr else f"reg_model_{b}.pkl"
        reg_model = RegressionModel(
            n_actions=obd.n_actions,
            len_list=obd.len_list,
            action_context=bandit_feedback_train["action_context"],
            base_model=base_model_dict[base_model](**hyperparams[base_model]),
            fitting_method=fitting_method,
        )
        # train regression model on logged bandit feedback data
        reg_model.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
            position=bandit_feedback_train["position"],
            action_dist=np.tile(
                action_dist_single_round, (bandit_feedback_train["n_rounds"], 1, 1)
            ),
        )
        with open(reg_model_path / model_file_name, "wb") as f:
            pickle.dump(
                reg_model,
                f,
            )
        # evaluate the estimation performance of the regression model by AUC and RCE
        estimated_rewards_by_reg_model = reg_model.predict(
            context=bandit_feedback_test["context"],
        )
        performance_reg_model_b = evaluate_reg_model(
            bandit_feedback=bandit_feedback_test,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        return performance_reg_model_b

    processed = Parallel(
        n_jobs=n_jobs,
        verbose=50,
    )([delayed(process)(i) for i in np.arange(n_runs)])
    # save performance of the regression model in './logs' directory.
    if not is_mrdr:
        performance_reg_model = {metric: dict() for metric in ["auc", "rce"]}
        for b, performance_reg_model_b in enumerate(processed):
            for metric, metric_value in performance_reg_model_b.items():
                performance_reg_model[metric][b] = metric_value
        DataFrame(performance_reg_model).describe().T.round(6).to_csv(
            log_path / "performance_reg_model.csv"
        )
