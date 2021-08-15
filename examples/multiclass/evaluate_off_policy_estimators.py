import argparse
import yaml
from pathlib import Path

import numpy as np
from pandas import DataFrame
from joblib import Parallel, delayed
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from obp.dataset import MultiClassToBanditReduction
from obp.ope import (
    RegressionModel,
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
    SelfNormalizedDoublyRobust,
    SwitchDoublyRobust,
    DoublyRobustWithShrinkage,
)


# hyperparameters of the regression model used in model dependent OPE estimators
with open("./conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

dataset_dict = dict(
    breast_cancer=load_breast_cancer(return_X_y=True),
    digits=load_digits(return_X_y=True),
    iris=load_iris(return_X_y=True),
    wine=load_wine(return_X_y=True),
)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

# compared OPE estimators
ope_estimators = [
    DirectMethod(),
    InverseProbabilityWeighting(),
    SelfNormalizedInverseProbabilityWeighting(),
    DoublyRobust(),
    SelfNormalizedDoublyRobust(),
    SwitchDoublyRobust(tau=1.0, estimator_name="switch-dr (tau=1)"),
    SwitchDoublyRobust(tau=100.0, estimator_name="switch-dr (tau=100)"),
    DoublyRobustWithShrinkage(lambda_=1.0, estimator_name="dr-os (lambda=1)"),
    DoublyRobustWithShrinkage(lambda_=100.0, estimator_name="dr-os (lambda=100)"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate off-policy estimators with multi-class classification data."
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="number of simulations in the experiment."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["breast_cancer", "digits", "iris", "wine"],
        required=True,
        help="the name of the multi-class classification dataset.",
    )
    parser.add_argument(
        "--eval_size",
        type=float,
        default=0.7,
        help="the proportion of the dataset to include in the evaluation split.",
    )
    parser.add_argument(
        "--base_model_for_behavior_policy",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for behavior policy, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--alpha_b",
        type=float,
        default=0.8,
        help="the ratio of a uniform random policy when constructing an behavior policy.",
    )
    parser.add_argument(
        "--base_model_for_evaluation_policy",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for evaluation policy, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--alpha_e",
        type=float,
        default=0.9,
        help="the ratio of a uniform random policy when constructing an evaluation policy.",
    )
    parser.add_argument(
        "--base_model_for_reg_model",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for regression model, logistic_regression, random_forest or lightgbm.",
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
    dataset_name = args.dataset_name
    eval_size = args.eval_size
    base_model_for_behavior_policy = args.base_model_for_behavior_policy
    alpha_b = args.alpha_b
    base_model_for_evaluation_policy = args.base_model_for_evaluation_policy
    alpha_e = args.alpha_e
    base_model_for_reg_model = args.base_model_for_reg_model
    n_jobs = args.n_jobs
    random_state = args.random_state
    np.random.seed(random_state)

    # load raw data
    X, y = dataset_dict[dataset_name]
    # convert the raw classification data into a logged bandit dataset
    dataset = MultiClassToBanditReduction(
        X=X,
        y=y,
        base_classifier_b=base_model_dict[base_model_for_behavior_policy](
            **hyperparams[base_model_for_behavior_policy]
        ),
        alpha_b=alpha_b,
        dataset_name=dataset_name,
    )

    def process(i: int):
        # split the original data into training and evaluation sets
        dataset.split_train_eval(eval_size=eval_size, random_state=i)
        # obtain logged bandit feedback generated by behavior policy
        bandit_feedback = dataset.obtain_batch_bandit_feedback(random_state=i)
        # obtain action choice probabilities by an evaluation policy
        action_dist = dataset.obtain_action_dist_by_eval_policy(
            base_classifier_e=base_model_dict[base_model_for_evaluation_policy](
                **hyperparams[base_model_for_evaluation_policy]
            ),
            alpha_e=alpha_e,
        )
        # calculate the ground-truth performance of the evaluation policy
        ground_truth_policy_value = dataset.calc_ground_truth_policy_value(
            action_dist=action_dist
        )
        # estimate the mean reward function of the evaluation set of multi-class classification data with ML model
        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=bandit_feedback["context"],
            action=bandit_feedback["action"],
            reward=bandit_feedback["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )
        # evaluate estimators' performances using relative estimation error (relative-ee)
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback,
            ope_estimators=ope_estimators,
        )
        relative_ee_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        return relative_ee_i

    processed = Parallel(
        n_jobs=n_jobs,
        verbose=50,
    )([delayed(process)(i) for i in np.arange(n_runs)])
    relative_ee_dict = {est.estimator_name: dict() for est in ope_estimators}
    for i, relative_ee_i in enumerate(processed):
        for (
            estimator_name,
            relative_ee_,
        ) in relative_ee_i.items():
            relative_ee_dict[estimator_name][i] = relative_ee_
    relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)

    print("=" * 45)
    print(f"random_state={random_state}")
    print("-" * 45)
    print(relative_ee_df[["mean", "std"]])
    print("=" * 45)

    # save results of the evaluation of off-policy estimators in './logs' directory.
    log_path = Path(f"./logs/{dataset_name}")
    log_path.mkdir(exist_ok=True, parents=True)
    relative_ee_df.to_csv(log_path / "relative_ee_of_ope_estimators.csv")
