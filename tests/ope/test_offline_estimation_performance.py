import numpy as np
from pandas import DataFrame
from joblib import Parallel, delayed
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import pytest
from dataclasses import dataclass
from obp.ope.estimators import BaseOffPolicyEstimator

from typing import Dict

from obp.dataset import (
    SyntheticBanditDataset,
    linear_behavior_policy,
    logistic_reward_function,
)
from obp.policy import IPWLearner
from obp.ope import (
    RegressionModel,
    OffPolicyEvaluation,
    InverseProbabilityWeighting,
    InverseProbabilityWeightingTuning,
    SelfNormalizedInverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
    DoublyRobustTuning,
    SelfNormalizedDoublyRobust,
    SwitchDoublyRobust,
    SwitchDoublyRobustTuning,
    DoublyRobustWithShrinkage,
    DoublyRobustWithShrinkageTuning,
)


# hyperparameters of the regression model used in model dependent OPE estimators
hyperparams = {
    "lightgbm": {
        "max_iter": 500,
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
}

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=HistGradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

offline_experiment_configurations = [
    (
        600,
        10,
        5,
        "logistic_regression",
        "logistic_regression",
    ),
    (
        300,
        3,
        2,
        "lightgbm",
        "lightgbm",
    ),
    (
        500,
        5,
        3,
        "random_forest",
        "random_forest",
    ),
    (
        500,
        3,
        5,
        "logistic_regression",
        "random_forest",
    ),
    (
        400,
        10,
        10,
        "lightgbm",
        "logistic_regression",
    ),
]


@dataclass
class RandomOffPolicyEstimator(BaseOffPolicyEstimator):
    """Estimate the policy value based on random predictions"""

    estimator_name: str = "random"

    def _estimate_round_rewards(
        self,
        action_dist: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        n_rounds = action_dist.shape[0]
        return np.random.uniform(size=n_rounds)

    def estimate_policy_value(
        self,
        action_dist: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy."""
        return self._estimate_round_rewards(action_dist=action_dist).mean()

    def estimate_policy_value_tensor(self, **kwargs) -> torch.Tensor:
        pass  # not used in this test

    def estimate_interval(self) -> Dict[str, float]:
        pass  # not used in this test


# compared OPE estimators
ope_estimators = [
    RandomOffPolicyEstimator(),
    DirectMethod(),
    InverseProbabilityWeighting(),
    InverseProbabilityWeightingTuning(
        lambdas=[100, 1000, np.inf], estimator_name="ipw (tuning)"
    ),
    SelfNormalizedInverseProbabilityWeighting(),
    DoublyRobust(),
    DoublyRobustTuning(lambdas=[100, 1000, np.inf], estimator_name="dr (tuning)"),
    SelfNormalizedDoublyRobust(),
    SwitchDoublyRobust(tau=1.0, estimator_name="switch-dr (tau=1)"),
    SwitchDoublyRobust(tau=100.0, estimator_name="switch-dr (tau=100)"),
    SwitchDoublyRobustTuning(
        taus=[100, 1000, np.inf], estimator_name="switch-dr (tuning)"
    ),
    DoublyRobustWithShrinkage(lambda_=1.0, estimator_name="dr-os (lambda=1)"),
    DoublyRobustWithShrinkage(lambda_=100.0, estimator_name="dr-os (lambda=100)"),
    DoublyRobustWithShrinkageTuning(
        lambdas=[100, 1000, np.inf], estimator_name="dr-os (tuning)"
    ),
]


@pytest.mark.parametrize(
    "n_rounds, n_actions, dim_context, base_model_for_evaluation_policy, base_model_for_reg_model",
    offline_experiment_configurations,
)
def test_offline_estimation_performance(
    n_rounds: int,
    n_actions: int,
    dim_context: int,
    base_model_for_evaluation_policy: str,
    base_model_for_reg_model: str,
) -> None:
    def process(i: int):
        # synthetic data generator
        dataset = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            reward_function=logistic_reward_function,
            behavior_policy_function=linear_behavior_policy,
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

    assert relative_ee_df_mean["random"] > relative_ee_df_mean["dm"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["ipw"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["ipw (tuning)"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["snipw"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["dr"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["dr (tuning)"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["sndr"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["switch-dr (tau=1)"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["switch-dr (tau=100)"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["switch-dr (tuning)"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["dr-os (lambda=1)"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["dr-os (lambda=100)"]
    assert relative_ee_df_mean["random"] > relative_ee_df_mean["dr-os (tuning)"]
