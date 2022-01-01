from dataclasses import dataclass
from typing import Dict

from joblib import delayed
from joblib import Parallel
import numpy as np
from pandas import DataFrame
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch

from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticBanditDataset
from obp.ope import BalancedInverseProbabilityWeighting
from obp.ope import DirectMethod
from obp.ope import DoublyRobust
from obp.ope import DoublyRobustTuning
from obp.ope import DoublyRobustWithShrinkage
from obp.ope import DoublyRobustWithShrinkageTuning
from obp.ope import InverseProbabilityWeighting
from obp.ope import InverseProbabilityWeightingTuning
from obp.ope import ImportanceWeightEstimator
from obp.ope import OffPolicyEvaluation
from obp.ope import PropensityScoreEstimator
from obp.ope import RegressionModel
from obp.ope import SelfNormalizedDoublyRobust
from obp.ope import SelfNormalizedInverseProbabilityWeighting
from obp.ope import SwitchDoublyRobust
from obp.ope import SwitchDoublyRobustTuning
from obp.ope.estimators import BaseOffPolicyEstimator
from obp.policy import IPWLearner

# hyperparameters of the regression model used in model dependent OPE estimators
hyperparams = {
    "lightgbm": {
        "n_estimators": 100,
        "learning_rate": 0.005,
        "max_depth": 5,
        "min_samples_leaf": 10,
        "random_state": 12345,
    },
    "logistic_regression": {
        "max_iter": 10000,
        "C": 1000,
        "random_state": 12345,
    },
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 5,
        "min_samples_leaf": 10,
        "random_state": 12345,
    },
    "svc": {"gamma": 2, "C": 5, "probability": True, "random_state": 12345},
}

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=GradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

offline_experiment_configurations = [
    (
        600,
        10,
        5,
        "logistic_regression",
        "logistic_regression",
        "logistic_regression",
    ),
    (
        300,
        3,
        2,
        "lightgbm",
        "lightgbm",
        "lightgbm",
    ),
    (
        500,
        5,
        3,
        "random_forest",
        "random_forest",
        "random_forest",
    ),
    (
        500,
        3,
        5,
        "logistic_regression",
        "random_forest",
        "random_forest",
    ),
    (
        800,
        10,
        10,
        "lightgbm",
        "logistic_regression",
        "logistic_regression",
    ),
]

bipw_model_configurations = {
    "bipw (random_forest raw)": dict(
        fitting_method="raw",
        base_model=RandomForestClassifier(**hyperparams["random_forest"]),
    ),
    "bipw (random_forest sample)": dict(
        fitting_method="sample",
        base_model=RandomForestClassifier(**hyperparams["random_forest"]),
    ),
    "bipw (svc raw)": dict(
        fitting_method="raw",
        base_model=SVC(**hyperparams["svc"]),
    ),
    "bipw (svc sample)": dict(
        fitting_method="sample",
        base_model=SVC(**hyperparams["svc"]),
    ),
}


@dataclass
class NaiveEstimator(BaseOffPolicyEstimator):
    """Estimate the policy value by just averaging observed rewards"""

    estimator_name: str = "naive"

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return reward

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy."""
        return self._estimate_round_rewards(reward=reward).mean()

    def estimate_policy_value_tensor(self, **kwargs) -> torch.Tensor:
        pass  # not used in this test

    def estimate_interval(self) -> Dict[str, float]:
        pass  # not used in this test


# compared OPE estimators
ope_estimators = [
    NaiveEstimator(),
    DirectMethod(),
    InverseProbabilityWeighting(),
    InverseProbabilityWeightingTuning(
        lambdas=[100, 1000, np.inf], estimator_name="ipw (tuning)"
    ),
    SelfNormalizedInverseProbabilityWeighting(),
    DoublyRobust(),
    DoublyRobustTuning(lambdas=[100, 1000, np.inf], estimator_name="dr (tuning)"),
    SelfNormalizedDoublyRobust(),
    SwitchDoublyRobust(lambda_=1.0, estimator_name="switch-dr (lambda=1)"),
    SwitchDoublyRobust(lambda_=100.0, estimator_name="switch-dr (lambda=100)"),
    SwitchDoublyRobustTuning(
        lambdas=[100, 1000, np.inf], estimator_name="switch-dr (tuning)"
    ),
    DoublyRobustWithShrinkage(lambda_=1.0, estimator_name="dr-os (lambda=1)"),
    DoublyRobustWithShrinkage(lambda_=100.0, estimator_name="dr-os (lambda=100)"),
    DoublyRobustWithShrinkageTuning(
        lambdas=[100, 1000, np.inf], estimator_name="dr-os (tuning)"
    ),
    InverseProbabilityWeighting(
        lambda_=100,
        estimator_name="cipw (estimated pscore)",
        use_estimated_pscore=True,
    ),
    SelfNormalizedInverseProbabilityWeighting(
        estimator_name="snipw (estimated pscore)", use_estimated_pscore=True
    ),
    DoublyRobust(estimator_name="dr (estimated pscore)", use_estimated_pscore=True),
    DoublyRobustWithShrinkage(
        lambda_=500,
        estimator_name="dr-os (estimated pscore)",
        use_estimated_pscore=True,
    ),
    BalancedInverseProbabilityWeighting(
        estimator_name="bipw (svc raw)", lambda_=np.inf
    ),
    BalancedInverseProbabilityWeighting(
        estimator_name="bipw (svc sample)", lambda_=np.inf
    ),
    BalancedInverseProbabilityWeighting(
        estimator_name="bipw (random_forest raw)", lambda_=np.inf
    ),
    BalancedInverseProbabilityWeighting(
        estimator_name="bipw (random_forest sample)", lambda_=np.inf
    ),
]


@pytest.mark.parametrize(
    "n_rounds, n_actions, dim_context, base_model_for_evaluation_policy, base_model_for_reg_model, base_model_for_treatment_model",
    offline_experiment_configurations,
)
def test_offline_estimation_performance(
    n_rounds: int,
    n_actions: int,
    dim_context: int,
    base_model_for_evaluation_policy: str,
    base_model_for_reg_model: str,
    base_model_for_treatment_model: str,
) -> None:
    def process(i: int):
        # synthetic data generator
        dataset = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            beta=-2.0,
            reward_function=logistic_reward_function,
            random_state=i,
        )
        # define evaluation policy using IPWLearner
        evaluation_policy = IPWLearner(
            n_actions=dataset.n_actions,
            base_classifier=base_model_dict[base_model_for_evaluation_policy](
                **hyperparams[base_model_for_evaluation_policy]
            ),
        )
        # sample new training and test sets of synthetic logged bandit feedback
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        # train the evaluation policy on the training set of the synthetic logged bandit feedback
        evaluation_policy.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
        )
        # predict the action decisions for the test set of the synthetic logged bandit feedback
        action_dist = evaluation_policy.predict(
            context=bandit_feedback_test["context"],
        )
        # estimate the mean reward function of the test set of synthetic bandit feedback with ML model
        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            action_context=dataset.action_context,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=bandit_feedback_test["context"],
            action=bandit_feedback_test["action"],
            reward=bandit_feedback_test["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=12345,
        )
        # fit propensity score estimators
        classification_model_for_action = PropensityScoreEstimator(
            len_list=1,
            n_actions=n_actions,
            base_model=base_model_dict[base_model_for_treatment_model](
                **hyperparams[base_model_for_treatment_model]
            ),
            calibration_cv=2,
        )
        estimated_pscore = classification_model_for_action.fit_predict(
            action=bandit_feedback_test["action"],
            position=bandit_feedback_test["position"],
            context=bandit_feedback_test["context"],
            n_folds=3,
            evaluate_model_performance=True,
            random_state=12345,
        )
        # fit importance weight estimators
        estimated_importance_weights_dict = {}
        for clf_name, clf_arguments in bipw_model_configurations.items():
            clf = ImportanceWeightEstimator(
                len_list=1,
                n_actions=n_actions,
                fitting_method=clf_arguments["fitting_method"],
                base_model=clf_arguments["base_model"],
            )
            estimated_importance_weights_dict[clf_name] = clf.fit_predict(
                action=bandit_feedback_test["action"],
                context=bandit_feedback_test["context"],
                action_dist=action_dist,
                position=bandit_feedback_test["position"],
                n_folds=2,
                evaluate_model_performance=False,
                random_state=12345,
            )
        # evaluate estimators' performances using relative estimation error (relative-ee)
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback_test,
            ope_estimators=ope_estimators,
        )
        relative_ee_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=dataset.calc_ground_truth_policy_value(
                expected_reward=bandit_feedback_test["expected_reward"],
                action_dist=action_dist,
            ),
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_importance_weights=estimated_importance_weights_dict,
        )

        return relative_ee_i

    n_runs = 10
    processed = Parallel(
        n_jobs=-1,
        verbose=0,
    )([delayed(process)(i) for i in np.arange(n_runs)])
    relative_ee_dict = {est.estimator_name: dict() for est in ope_estimators}
    for i, relative_ee_i in enumerate(processed):
        for (
            estimator_name,
            relative_ee_,
        ) in relative_ee_i.items():
            relative_ee_dict[estimator_name][i] = relative_ee_
    relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)
    relative_ee_df_mean = relative_ee_df["mean"]

    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["dm"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["ipw"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["ipw (tuning)"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["snipw"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["dr"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["dr (tuning)"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["sndr"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["switch-dr (lambda=1)"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["switch-dr (lambda=100)"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["switch-dr (tuning)"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["dr-os (lambda=1)"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["dr-os (lambda=100)"]
    assert relative_ee_df_mean["naive"] > relative_ee_df_mean["dr-os (tuning)"]
    # test estimated_pscore and bipw
    estimated_pscore_and_bipw_estimators = [
        "cipw (estimated pscore)",
        "snipw (estimated pscore)",
        "dr (estimated pscore)",
        "dr-os (estimated pscore)",
        "bipw (svc raw)",
        "bipw (svc sample)",
        "bipw (random_forest raw)",
        "bipw (random_forest sample)",
    ]
    for estimator_name in estimated_pscore_and_bipw_estimators:
        assert (
            relative_ee_df_mean["naive"] > relative_ee_df_mean[estimator_name]
        ), f"{estimator_name} is worse than naive estimator"
    #     print(estimator_name, relative_ee_df_mean[estimator_name])
    # print(relative_ee_df_mean["naive"])
