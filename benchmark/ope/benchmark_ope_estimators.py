from logging import getLogger
from pathlib import Path
import time
import warnings

import hydra
import numpy as np
from omegaconf import DictConfig
from pandas import DataFrame
import pingouin as pg
from pyieoe.evaluator import InterpretableOPEEvaluator
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier as LightGBM
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from obp.dataset import OpenBanditDataset
from obp.ope import DirectMethod
from obp.ope import DoublyRobust
from obp.ope import DoublyRobustWithShrinkageTuning
from obp.ope import InverseProbabilityWeighting
from obp.ope import SelfNormalizedDoublyRobust
from obp.ope import SelfNormalizedInverseProbabilityWeighting
from obp.ope import SwitchDoublyRobustTuning
from obp.policy import BernoulliTS
from obp.policy import Random


logger = getLogger(__name__)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

reg_model_dict = dict(
    logistic_regression=LogisticRegression,
    random_forest=RandomForest,
    lightgbm=LightGBM,
)


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    logger.info(f"The current working directory is {Path().cwd()}")
    start_time = time.time()
    logger.info("initializing experimental condition..")

    # compared ope estimators
    lambdas = list(dict(cfg.estimator_hyperparams)["lambdas"])
    ope_estimators = [
        InverseProbabilityWeighting(estimator_name="IPW"),
        SelfNormalizedInverseProbabilityWeighting(estimator_name="SNIPW"),
        DirectMethod(estimator_name="DM"),
        DoublyRobust(estimator_name="DR"),
        SelfNormalizedDoublyRobust(estimator_name="SNDR"),
        SwitchDoublyRobustTuning(lambdas=lambdas, estimator_name="Switch-DR"),
        DoublyRobustWithShrinkageTuning(lambdas=lambdas, estimator_name="DRos"),
    ]

    # configurations
    n_seeds = cfg.setting.n_seeds
    sample_size = cfg.setting.sample_size
    reg_model = cfg.setting.reg_model
    campaign = cfg.setting.campaign
    behavior_policy = cfg.setting.behavior_policy
    test_size = cfg.setting.test_size
    is_timeseries_split = cfg.setting.is_timeseries_split
    n_folds = cfg.setting.n_folds
    obd_path = (
        Path().cwd().parents[5] / "open_bandit_dataset"
        if cfg.setting.is_full_obd
        else None
    )
    random_state = cfg.setting.random_state
    np.random.seed(random_state)

    # define dataset
    dataset_ts = OpenBanditDataset(
        behavior_policy="bts", campaign=campaign, data_path=obd_path
    )
    dataset_ur = OpenBanditDataset(
        behavior_policy="random", campaign=campaign, data_path=obd_path
    )

    # prepare logged bandit feedback and evaluation policies
    if behavior_policy == "random":
        if is_timeseries_split:
            bandit_feedback_ur = dataset_ur.obtain_batch_bandit_feedback(
                test_size=test_size,
                is_timeseries_split=True,
            )[0]
        else:
            bandit_feedback_ur = dataset_ur.obtain_batch_bandit_feedback()
        bandit_feedbacks = [bandit_feedback_ur]
        # obtain the ground-truth policy value
        ground_truth_ts = OpenBanditDataset.calc_on_policy_policy_value_estimate(
            behavior_policy="bts",
            campaign=campaign,
            data_path=obd_path,
            test_size=test_size,
            is_timeseries_split=is_timeseries_split,
        )
        # obtain action choice probabilities and define evaluation policies
        policy_ts = BernoulliTS(
            n_actions=dataset_ts.n_actions,
            len_list=dataset_ts.len_list,
            random_state=random_state,
            is_zozotown_prior=True,
            campaign=campaign,
        )
        action_dist_ts = policy_ts.compute_batch_action_dist(n_rounds=1000000)
        evaluation_policies = [(ground_truth_ts, action_dist_ts)]
    else:
        if is_timeseries_split:
            bandit_feedback_ts = dataset_ts.obtain_batch_bandit_feedback(
                test_size=test_size,
                is_timeseries_split=True,
            )[0]
        else:
            bandit_feedback_ts = dataset_ts.obtain_batch_bandit_feedback()
        bandit_feedbacks = [bandit_feedback_ts]
        # obtain the ground-truth policy value
        ground_truth_ur = OpenBanditDataset.calc_on_policy_policy_value_estimate(
            behavior_policy="random",
            campaign=campaign,
            data_path=obd_path,
            test_size=test_size,
            is_timeseries_split=is_timeseries_split,
        )
        # obtain action choice probabilities and define evaluation policies
        policy_ur = Random(
            n_actions=dataset_ur.n_actions,
            len_list=dataset_ur.len_list,
            random_state=random_state,
        )
        action_dist_ur = policy_ur.compute_batch_action_dist(n_rounds=1000000)
        evaluation_policies = [(ground_truth_ur, action_dist_ur)]

    # regression models used in ope estimators
    hyperparams = dict(cfg.reg_model_hyperparams)[reg_model]
    regression_models = [reg_model_dict[reg_model](**hyperparams)]

    # define an evaluator class
    evaluator = InterpretableOPEEvaluator(
        random_states=np.arange(n_seeds),
        bandit_feedbacks=bandit_feedbacks,
        evaluation_policies=evaluation_policies,
        ope_estimators=ope_estimators,
        regression_models=regression_models,
    )

    # conduct an evaluation of OPE experiment
    logger.info("experiment started")
    _ = evaluator.estimate_policy_value(sample_size=sample_size, n_folds_=n_folds)
    # calculate statistics
    mean = evaluator.calculate_mean(root=True)
    mean_scaled = evaluator.calculate_mean(scale=True, root=True)

    # save results of the evaluation of off-policy estimators
    log_path = Path("./outputs")
    log_path.mkdir(exist_ok=True, parents=True)
    # save root mse
    root_mse_df = DataFrame()
    root_mse_df["estimator"] = list(mean.keys())
    root_mse_df["mean"] = list(mean.values())
    root_mse_df["mean(scaled)"] = list(mean_scaled.values())
    root_mse_df.to_csv(log_path / "root_mse.csv")
    # conduct pairwise t-tests
    se_df = DataFrame(evaluator.calculate_squared_error())
    se_df = DataFrame(se_df.stack()).reset_index(1)
    se_df.rename(columns={"level_1": "estimators", 0: "se"}, inplace=True)
    nonparam_ttests = (
        pg.pairwise_ttests(
            data=se_df,
            dv="se",
            parametric=False,
            between="estimators",
        )
        .round(4)
        .drop(["Contrast", "Parametric", "Paired"], axis=1)
    )
    nonparam_ttests.to_csv(log_path / "nonparam_ttests.csv")
    # save reg model metrics
    DataFrame(evaluator.reg_model_metrics).describe().to_csv(
        log_path / "reg_model_metrics.csv"
    )
    # print result
    print(root_mse_df)
    experiment = f"{campaign}-{behavior_policy}-{sample_size}"
    elapsed_time = np.round((time.time() - start_time) / 60, 2)
    logger.info(f"finish experiment {experiment} in {elapsed_time}min")


if __name__ == "__main__":
    main()
