import argparse
import yaml
from pathlib import Path

from pandas import DataFrame
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from obp.dataset import (
    SyntheticBanditDataset,
    linear_behavior_policy,
    logistic_reward_function,
)
from obp.policy import IPWLearner, NNPolicyLearner, Random
from obp.ope import (
    RegressionModel,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
    SelfNormalizedDoublyRobust,
    DoublyRobustWithShrinkage,
)


# hyperparameters of the regression model used in model dependent OPE estimators
with open("./conf/hyperparams.yaml", "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

ope_estimator_dict = dict(
    dm=DirectMethod(),
    ipw=InverseProbabilityWeighting(),
    snipw=SelfNormalizedInverseProbabilityWeighting(),
    dr=DoublyRobust(),
    sndr=SelfNormalizedDoublyRobust(),
    dros=DoublyRobustWithShrinkage(lambda_=1.0),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate off-policy estimators with synthetic bandit data."
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=10000,
        help="number of rounds for synthetic bandit feedback.",
    )
    parser.add_argument(
        "--n_actions",
        type=int,
        default=10,
        help="number of actions for synthetic bandit feedback.",
    )
    parser.add_argument(
        "--dim_context",
        type=int,
        default=5,
        help="dimensions of context vectors characterizing each round.",
    )
    parser.add_argument(
        "--base_model_for_evaluation_policy",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for evaluation policy, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--base_model_for_reg_model",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for regression model, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--ope_estimator",
        type=str,
        choices=["dm", "ipw", "snipw", "dr", "sndr", "dros"],
        required=True,
        help="OPE estimator for NNPolicyLearner, dm, ipw, snipw, dr, sndr, or dros.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=100,
        help="the size of hidden layers",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="the number of hidden layers",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["identity", "logistic", "tanh", "relu"],
        default="relu",
        help="activation function for the NN Policy Learner",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["lbfgs", "sgd", "adam"],
        default="adam",
        help="optimizer for the NN Policy Learner",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="batch size for the NN Policy Learner",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="use early stopping in training of the NN Policy Learner",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations
    n_rounds = args.n_rounds
    n_actions = args.n_actions
    dim_context = args.dim_context
    base_model_for_evaluation_policy = args.base_model_for_evaluation_policy
    base_model_for_reg_model = args.base_model_for_reg_model
    ope_estimator = args.ope_estimator
    n_hidden = args.n_hidden
    n_layers = args.n_layers
    activation = args.activation
    solver = args.solver
    batch_size = args.batch_size if args.batch_size else "auto"
    early_stopping = args.early_stopping
    random_state = args.random_state

    # synthetic data generator
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_function=logistic_reward_function,
        behavior_policy_function=linear_behavior_policy,
        random_state=random_state,
    )
    # sample new training and test sets of synthetic logged bandit feedback
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    # estimate the mean reward function of the train set of synthetic bandit feedback with ML model
    regression_model = RegressionModel(
        n_actions=dataset.n_actions,
        action_context=dataset.action_context,
        base_model=base_model_dict[base_model_for_reg_model](
            **hyperparams[base_model_for_reg_model]
        ),
    )
    estimated_rewards_by_reg_model = regression_model.fit_predict(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        n_folds=3,  # 3-fold cross-fitting
        random_state=random_state,
    )
    # define random evaluation policy
    random_policy = Random(n_actions=dataset.n_actions, random_state=random_state)
    # define evaluation policy using IPWLearner
    ipw_learner = IPWLearner(
        n_actions=dataset.n_actions,
        base_classifier=base_model_dict[base_model_for_evaluation_policy](
            **hyperparams[base_model_for_evaluation_policy]
        ),
    )
    # define evaluation policy using NNPolicyLearner
    nn_policy_learner = NNPolicyLearner(
        n_actions=dataset.n_actions,
        dim_context=dim_context,
        off_policy_objective=ope_estimator_dict[
            ope_estimator
        ].estimate_policy_value_tensor,
        hidden_layer_size=tuple((n_hidden for _ in range(n_layers))),
        activation=activation,
        solver=solver,
        batch_size=batch_size,
        early_stopping=early_stopping,
        random_state=random_state,
    )
    # train the evaluation policy on the training set of the synthetic logged bandit feedback
    ipw_learner.fit(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        pscore=bandit_feedback_train["pscore"],
    )
    nn_policy_learner.fit(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        pscore=bandit_feedback_train["pscore"],
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    # predict the action decisions for the test set of the synthetic logged bandit feedback
    random_action_dist = random_policy.compute_batch_action_dist(n_rounds=n_rounds)
    ipw_learner_action_dist = ipw_learner.predict(
        context=bandit_feedback_test["context"],
    )
    nn_policy_learner_action_dist = nn_policy_learner.predict_proba(
        context=bandit_feedback_test["context"],
    )

    # evaluate learners' performances using ground-truth polocy values
    random_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=random_action_dist,
    )
    ipw_learner_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=ipw_learner_action_dist,
    )
    nn_policy_learner_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=nn_policy_learner_action_dist,
    )

    policy_value_df = DataFrame(
        [
            [random_policy_value],
            [ipw_learner_policy_value],
            [nn_policy_learner_policy_value],
        ],
        columns=["policy value"],
        index=[
            "random_policy",
            "ipw_learner",
            f"nn_policy_learner (with {ope_estimator})",
        ],
    ).round(6)
    print("=" * 45)
    print(f"random_state={random_state}")
    print("-" * 45)
    print(policy_value_df)
    print("=" * 45)

    # save results of the evaluation of off-policy learners in './logs' directory.
    log_path = Path("./logs")
    log_path.mkdir(exist_ok=True, parents=True)
    policy_value_df.to_csv(log_path / "policy_value_of_off_policy_learners.csv")
