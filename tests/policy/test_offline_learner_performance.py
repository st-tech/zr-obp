from dataclasses import dataclass
from typing import Optional
from typing import Tuple

from joblib import delayed
from joblib import Parallel
import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from obp.dataset import linear_behavior_policy
from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticBanditDataset
from obp.policy import IPWLearner
from obp.policy import NNPolicyLearner
from obp.policy import QLearner
from obp.policy.base import BaseOfflinePolicyLearner


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
}

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=GradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

# n_rounds, n_actions, dim_context, base_model_for_evaluation_policy, base_model_for_reg_model
offline_experiment_configurations = [
    (
        600,
        10,
        5,
        "logistic_regression",
        "logistic_regression",
    ),
    (
        450,
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
        800,
        10,
        10,
        "lightgbm",
        "logistic_regression",
    ),
]


@dataclass
class RandomPolicy(BaseOfflinePolicyLearner):
    def __post_init__(self) -> None:
        super().__post_init__()

    def fit(self):
        raise NotImplementedError

    def predict(self, context: np.ndarray) -> np.ndarray:

        n_rounds = context.shape[0]
        action_dist = np.random.rand(n_rounds, self.n_actions, self.len_list)
        return action_dist


@dataclass
class UniformSampleWeightLearner(BaseOfflinePolicyLearner):

    base_classifier: Optional[ClassifierMixin] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.base_classifier is None:
            self.base_classifier = LogisticRegression(random_state=12345)
        else:
            if not is_classifier(self.base_classifier):
                raise ValueError("`base_classifier` must be a classifier")
        self.base_classifier_list = [
            clone(self.base_classifier) for _ in np.arange(self.len_list)
        ]

    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        return context, (reward / pscore), action

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> None:

        if pscore is None:
            n_actions = np.int32(action.max() + 1)
            pscore = np.ones_like(action) / n_actions
        if position is None or self.len_list == 1:
            position = np.zeros_like(action, dtype=int)

        for pos_ in np.arange(self.len_list):
            X, sample_weight, y = self._create_train_data_for_opl(
                context=context[position == pos_],
                action=action[position == pos_],
                reward=reward[position == pos_],
                pscore=pscore[position == pos_],
            )
            self.base_classifier_list[pos_].fit(X=X, y=y)

    def predict(self, context: np.ndarray) -> np.ndarray:

        n_rounds = context.shape[0]
        action_dist = np.zeros((n_rounds, self.n_actions, self.len_list))
        for pos_ in np.arange(self.len_list):
            predicted_actions_at_position = self.base_classifier_list[pos_].predict(
                context
            )
            action_dist[
                np.arange(n_rounds),
                predicted_actions_at_position,
                np.ones(n_rounds, dtype=int) * pos_,
            ] += 1
        return action_dist


@pytest.mark.parametrize(
    "n_rounds, n_actions, dim_context, base_model_for_evaluation_policy, base_model_for_reg_model",
    offline_experiment_configurations,
)
def test_offline_policy_learner_performance(
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
        # sample new training and test sets of synthetic logged bandit data
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

        # defining policy learners
        ipw_policy = IPWLearner(
            n_actions=dataset.n_actions,
            base_classifier=base_model_dict[base_model_for_evaluation_policy](
                **hyperparams[base_model_for_evaluation_policy]
            ),
        )
        q_policy = QLearner(
            n_actions=dataset.n_actions,
            base_model=base_model_dict[base_model_for_evaluation_policy](
                **hyperparams[base_model_for_evaluation_policy]
            ),
        )
        nn_policy = NNPolicyLearner(
            n_actions=dataset.n_actions,
            dim_context=dim_context,
            off_policy_objective="ipw",
        )
        # baseline method 1. RandomPolicy
        random_policy = RandomPolicy(n_actions=dataset.n_actions)
        # baseline method 2. UniformSampleWeightLearner
        uniform_sample_weight_policy = UniformSampleWeightLearner(
            n_actions=dataset.n_actions,
            base_classifier=base_model_dict[base_model_for_evaluation_policy](
                **hyperparams[base_model_for_evaluation_policy]
            ),
        )

        # policy training
        ipw_policy.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
        )
        q_policy.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
        )
        nn_policy.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
        )
        uniform_sample_weight_policy.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
        )

        # prediction/making decisions
        ipw_action_dist = ipw_policy.predict(
            context=bandit_feedback_test["context"],
        )
        q_action_dist = q_policy.predict(
            context=bandit_feedback_test["context"],
        )
        nn_action_dist = nn_policy.predict(
            context=bandit_feedback_test["context"],
        )
        random_action_dist = random_policy.predict(
            context=bandit_feedback_test["context"],
        )
        uniform_sample_weight_action_dist = uniform_sample_weight_policy.predict(
            context=bandit_feedback_test["context"],
        )

        # evaluation
        gt_ipw_learner = dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=ipw_action_dist,
        )
        gt_q_learner = dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=q_action_dist,
        )
        gt_nn_learner = dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=nn_action_dist,
        )
        gt_random_policy = dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=random_action_dist,
        )
        gt_uniform_sample_weight_learner = dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=uniform_sample_weight_action_dist,
        )

        return (
            gt_ipw_learner,
            gt_q_learner,
            gt_nn_learner,
            gt_random_policy,
            gt_uniform_sample_weight_learner,
        )

    n_runs = 10
    processed = Parallel(
        n_jobs=-1,
        verbose=0,
    )([delayed(process)(i) for i in np.arange(n_runs)])
    list_gt_ipw = list()
    list_gt_q = list()
    list_gt_nn = list()
    list_gt_random = list()
    list_gt_unif_ipw = list()
    for i, gt_policy_values in enumerate(processed):
        gt_ipw, gt_q, gt_nn, gt_random, gt_unif_ipw = gt_policy_values
        list_gt_ipw.append(gt_ipw)
        list_gt_q.append(gt_q)
        list_gt_nn.append(gt_nn)
        list_gt_random.append(gt_random)
        list_gt_unif_ipw.append(gt_unif_ipw)

    # baseline learner performance
    print(f"Performance of Random is {np.mean(list_gt_random)}")
    print(
        f"Performance of IPWLearner with Uniform Weight is {np.mean(list_gt_unif_ipw)}"
    )
    # ipw learner performance
    print(f"Performance of IPWLearner is {np.mean(list_gt_ipw)}")
    assert np.mean(list_gt_ipw) > np.mean(list_gt_random)
    assert np.mean(list_gt_ipw) > np.mean(list_gt_unif_ipw)
    # q learner performance
    print(f"Performance of QLearner is {np.mean(list_gt_q)}")
    assert np.mean(list_gt_q) > np.mean(list_gt_random)
    assert np.mean(list_gt_q) > np.mean(list_gt_unif_ipw)
    # nn policy learner performance
    print(f"Performance of NNPolicyLearner is {np.mean(list_gt_nn)}")
    assert np.mean(list_gt_nn) > np.mean(list_gt_random)
    assert np.mean(list_gt_nn) > np.mean(list_gt_unif_ipw)
