import argparse
from pathlib import Path
import yaml

import numpy as np
from pandas import DataFrame
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from custom_dataset import OBDWithInteractionFeatures
from obp.policy import IPWLearner
from obp.ope import InverseProbabilityWeighting

# hyperparameters of the regression model used in model dependent OPE estimators
with open("./conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run evaluation policy selection.")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="number of bootstrap sampling in the experiment.",
    )
    parser.add_argument(
        "--context_set",
        type=str,
        choices=["1", "2"],
        required=True,
        help="context sets for contextual bandit policies.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base model for a evaluation policy to be evaluated",
    )
    parser.add_argument(
        "--behavior_policy",
        type=str,
        choices=["bts", "random"],
        default="random",
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
        default=0.5,
        help="the proportion of the dataset to include in the test split.",
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
    context_set = args.context_set
    base_model = args.base_model
    behavior_policy = args.behavior_policy
    campaign = args.campaign
    test_size = args.test_size
    n_jobs = args.n_jobs
    random_state = args.random_state
    np.random.seed(random_state)
    data_path = Path("../open_bandit_dataset")

    # define a dataset class
    obd = OBDWithInteractionFeatures(
        behavior_policy=behavior_policy,
        campaign=campaign,
        data_path=data_path,
        context_set=context_set,
    )
    # define a counterfactual policy based on IPWLearner
    counterfactual_policy = IPWLearner(
        base_model=base_model_dict[base_model](**hyperparams[base_model]),
        n_actions=obd.n_actions,
        len_list=obd.len_list,
    )
    policy_name = f"{base_model}_{context_set}"

    # ground-truth policy value of the Bernoulli TS policy (the current best policy) in the test set
    # , which is the empirical mean of the factual (observed) rewards (on-policy estimation)
    ground_truth = obd.calc_on_policy_policy_value_estimate(
        behavior_policy="bts",
        campaign=campaign,
        data_path=data_path,
        test_size=test_size,
        is_timeseries_split=True,
    )

    def process(b: int):
        # sample bootstrap from batch logged bandit feedback
        boot_bandit_feedback = obd.sample_bootstrap_bandit_feedback(
            test_size=test_size, is_timeseries_split=True, random_state=b
        )
        # train an evaluation on the training set of the logged bandit feedback data
        action_dist = counterfactual_policy.fit(
            context=boot_bandit_feedback["context"],
            action=boot_bandit_feedback["action"],
            reward=boot_bandit_feedback["reward"],
            pscore=boot_bandit_feedback["pscore"],
            position=boot_bandit_feedback["position"],
        )
        # make action selections (predictions)
        action_dist = counterfactual_policy.predict(
            context=boot_bandit_feedback["context_test"]
        )
        # estimate the policy value of a given counterfactual algorithm by the three OPE estimators.
        ipw = InverseProbabilityWeighting()
        return ipw.estimate_policy_value(
            reward=boot_bandit_feedback["reward_test"],
            action=boot_bandit_feedback["action_test"],
            position=boot_bandit_feedback["position_test"],
            pscore=boot_bandit_feedback["pscore_test"],
            action_dist=action_dist,
        )

    processed = Parallel(
        backend="multiprocessing",
        n_jobs=n_jobs,
        verbose=50,
    )([delayed(process)(i) for i in np.arange(n_runs)])

    # save counterfactual policy evaluation results in `./logs` directory
    ope_results = np.zeros((n_runs, 2))
    for b, estimated_policy_value_b in enumerate(processed):
        ope_results[b, 0] = estimated_policy_value_b
        ope_results[b, 1] = estimated_policy_value_b / ground_truth
    save_path = Path("./logs") / behavior_policy / campaign
    save_path.mkdir(exist_ok=True, parents=True)
    DataFrame(
        ope_results, columns=["policy_value", "relative_policy_value"]
    ).describe().round(6).to_csv(save_path / f"{policy_name}.csv")
