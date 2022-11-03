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
from obp.dataset import SyntheticBanditDatasetWithActionEmbeds
from obp.ope import BalancedInverseProbabilityWeighting
from obp.ope import DirectMethod
from obp.ope import DoublyRobustTuning
from obp.ope import DoublyRobustWithShrinkageTuning
from obp.ope import ImportanceWeightEstimator
from obp.ope import InverseProbabilityWeightingTuning
from obp.ope import MarginalizedInverseProbabilityWeighting
from obp.ope import OffPolicyEvaluation
from obp.ope import PropensityScoreEstimator
from obp.ope import RegressionModel
from obp.ope import SelfNormalizedDoublyRobust
from obp.ope import SelfNormalizedInverseProbabilityWeighting
from obp.ope import SelfNormalizedMarginalizedInverseProbabilityWeighting
from obp.ope import SubGaussianDoublyRobustTuning
from obp.ope import SubGaussianInverseProbabilityWeightingTuning
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
        1000,
        10,
        5,
        "logistic_regression",
        "logistic_regression",
        "logistic_regression",
    ),
    (
        600,
        10,
        10,
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
]

bipw_model_configurations = {
    "bipw (random_forest sample)": dict(
        fitting_method="sample",
        base_model=RandomForestClassifier(**hyperparams["random_forest"]),
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
    InverseProbabilityWeightingTuning(
        lambdas=[100, 500, 1000, 5000, np.inf],
        tuning_method="mse",
        estimator_name="ipw (tuning-mse)",
    ),
    InverseProbabilityWeightingTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
        tuning_method="slope",
        estimator_name="ipw (tuning-slope)",
    ),
    SubGaussianInverseProbabilityWeightingTuning(
        lambdas=[0.0001, 0.01],
        tuning_method="mse",
        estimator_name="sg-ipw (tuning-mse)",
    ),
    SelfNormalizedInverseProbabilityWeighting(),
    DoublyRobustTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
        tuning_method="mse",
        estimator_name="dr (tuning-mse)",
    ),
    DoublyRobustTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
        tuning_method="slope",
        estimator_name="dr (tuning-slope)",
    ),
    SelfNormalizedDoublyRobust(),
    SwitchDoublyRobustTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
        tuning_method="mse",
        estimator_name="switch-dr (tuning-mse)",
    ),
    SwitchDoublyRobustTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
        tuning_method="slope",
        estimator_name="switch-dr (tuning-slope)",
    ),
    DoublyRobustWithShrinkageTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
        tuning_method="mse",
        estimator_name="dr-os (tuning-mse)",
    ),
    DoublyRobustWithShrinkageTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
        tuning_method="slope",
        estimator_name="dr-os (tuning-slope)",
    ),
    SubGaussianDoublyRobustTuning(
        lambdas=[0.005, 0.01, 0.05, 0.1, 0.5],
        tuning_method="mse",
        estimator_name="sg-dr (tuning-mse)",
    ),
    SubGaussianDoublyRobustTuning(
        lambdas=[0.005, 0.01, 0.05, 0.1, 0.5],
        tuning_method="slope",
        estimator_name="sg-dr (tuning-slope)",
    ),
    InverseProbabilityWeightingTuning(
        lambdas=[10, 100, 1000],
        estimator_name="cipw (estimated pscore)",
        use_estimated_pscore=True,
    ),
    SelfNormalizedInverseProbabilityWeighting(
        estimator_name="snipw (estimated pscore)", use_estimated_pscore=True
    ),
    DoublyRobustTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
        estimator_name="dr (estimated pscore)",
        use_estimated_pscore=True,
    ),
    DoublyRobustWithShrinkageTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, np.inf],
        estimator_name="dr-os (estimated pscore)",
        use_estimated_pscore=True,
    ),
    BalancedInverseProbabilityWeighting(
        estimator_name="bipw (svc sample)", lambda_=100
    ),
    BalancedInverseProbabilityWeighting(
        estimator_name="bipw (random_forest sample)", lambda_=100
    ),
]


@pytest.mark.parametrize(
    "n_rounds, n_actions, dim_context, base_model_for_iw_estimator, base_model_for_reg_model, base_model_for_pscore_estimator",
    offline_experiment_configurations,
)
def test_offline_estimation_performance(
    n_rounds: int,
    n_actions: int,
    dim_context: int,
    base_model_for_iw_estimator: str,
    base_model_for_reg_model: str,
    base_model_for_pscore_estimator: str,
) -> None:
    def process(i: int):
        # synthetic data generator
        dataset = SyntheticBanditDatasetWithActionEmbeds(
            n_actions=n_actions,
            dim_context=dim_context,
            beta=3.0,
            n_cat_dim=3,
            n_cat_per_dim=5,
            reward_function=logistic_reward_function,
            random_state=i,
        )
        # define evaluation policy using IPWLearner
        evaluation_policy = IPWLearner(
            n_actions=dataset.n_actions,
            base_classifier=base_model_dict[base_model_for_iw_estimator](
                **hyperparams[base_model_for_iw_estimator]
            ),
        )
        # sample new training and test sets of synthetic logged bandit data
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        # train the evaluation policy on the training set of the synthetic logged bandit data
        evaluation_policy.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
        )
        # predict the action decisions for the test set of the synthetic logged bandit data
        action_dist = evaluation_policy.predict_proba(
            context=bandit_feedback_test["context"],
        )
        # estimate the reward function of the test set of synthetic bandit feedback with ML model
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
            n_folds=2,
            random_state=12345,
        )
        # fit propensity score estimators
        pscore_estimator = PropensityScoreEstimator(
            len_list=1,
            n_actions=n_actions,
            base_model=base_model_dict[base_model_for_pscore_estimator](
                **hyperparams[base_model_for_pscore_estimator]
            ),
            calibration_cv=3,
        )
        estimated_pscore = pscore_estimator.fit_predict(
            action=bandit_feedback_test["action"],
            position=bandit_feedback_test["position"],
            context=bandit_feedback_test["context"],
            n_folds=3,
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
            ope_estimators=ope_estimators
            + [
                MarginalizedInverseProbabilityWeighting(
                    n_actions=n_actions, estimator_name="mipw"
                ),
                MarginalizedInverseProbabilityWeighting(
                    n_actions=n_actions,
                    embedding_selection_method="greedy",
                    estimator_name="mipw (greedy selection)",
                ),
                SelfNormalizedMarginalizedInverseProbabilityWeighting(
                    n_actions=n_actions, estimator_name="snmipw"
                ),
            ],
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
            action_embed=bandit_feedback_test["action_embed"],
            pi_b=bandit_feedback_test["pi_b"],
            metric="relative-ee",
        )

        return relative_ee_i

    n_runs = 20
    processed = Parallel(
        n_jobs=-1,
        verbose=0,
    )([delayed(process)(i) for i in np.arange(n_runs)])
    metric_dict = {est.estimator_name: dict() for est in ope_estimators}
    metric_dict.update(
        {"mipw": dict(), "mipw (greedy selection)": dict(), "snmipw": dict()}
    )
    for i, relative_ee_i in enumerate(processed):
        for (
            estimator_name,
            relative_ee_,
        ) in relative_ee_i.items():
            metric_dict[estimator_name][i] = relative_ee_
    relative_ee_df = DataFrame(metric_dict).describe().T.round(6)
    relative_ee_df_mean = relative_ee_df["mean"]

    tested_estimators = [
        "dm",
        "ipw (tuning-mse)",
        "ipw (tuning-slope)",
        "sg-ipw (tuning-mse)",
        "snipw",
        "dr (tuning-mse)",
        "dr (tuning-slope)",
        "sndr",
        "switch-dr (tuning-mse)",
        "switch-dr (tuning-slope)",
        "dr-os (tuning-mse)",
        "dr-os (tuning-slope)",
        "sg-dr (tuning-mse)",
        "sg-dr (tuning-slope)",
        "cipw (estimated pscore)",
        "snipw (estimated pscore)",
        "dr (estimated pscore)",
        "dr-os (estimated pscore)",
        "bipw (svc sample)",
        "bipw (random_forest sample)",
        "mipw",
        "mipw (greedy selection)",
        "snmipw",
    ]
    for estimator_name in tested_estimators:
        assert (
            relative_ee_df_mean[estimator_name] / relative_ee_df_mean["naive"] < 1.5
        ), f"{estimator_name} is significantly worse than naive (on-policy) estimator"
