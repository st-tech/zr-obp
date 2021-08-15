from typing import Tuple, Union, Optional

import numpy as np
from joblib import Parallel, delayed
import pytest
from dataclasses import dataclass
from obp.dataset import (
    SyntheticContinuousBanditDataset,
    linear_reward_funcion_continuous,
    linear_behavior_policy_continuous,
)
from obp.policy import BaseContinuousOfflinePolicyLearner, ContinuousNNPolicyLearner


# n_rounds, dim_context, action_noise, reward_noise, min_action_value, max_action_value, pg_method, bandwidth
offline_experiment_configurations = [
    (
        1500,
        10,
        1.0,
        1.0,
        -10.0,
        10.0,
        "dpg",
        None,
    ),
    (
        2000,
        5,
        1.0,
        1.0,
        0.0,
        100.0,
        "dpg",
        None,
    ),
]


@dataclass
class RandomPolicy(BaseContinuousOfflinePolicyLearner):
    output_space: Tuple[Union[int, float], Union[int, float]] = None

    def fit(self):
        raise NotImplementedError

    def predict(self, context: np.ndarray) -> np.ndarray:

        n_rounds = context.shape[0]
        predicted_actions = np.random.uniform(
            self.output_space[0], self.output_space[1], size=n_rounds
        )
        return predicted_actions


@pytest.mark.parametrize(
    "n_rounds, dim_context, action_noise, reward_noise, min_action_value, max_action_value, pg_method, bandwidth",
    offline_experiment_configurations,
)
def test_offline_nn_policy_learner_performance(
    n_rounds: int,
    dim_context: int,
    action_noise: float,
    reward_noise: float,
    min_action_value: float,
    max_action_value: float,
    pg_method: str,
    bandwidth: Optional[float],
) -> None:
    def process(i: int):
        # synthetic data generator
        dataset = SyntheticContinuousBanditDataset(
            dim_context=dim_context,
            action_noise=action_noise,
            reward_noise=reward_noise,
            min_action_value=min_action_value,
            max_action_value=max_action_value,
            reward_function=linear_reward_funcion_continuous,
            behavior_policy_function=linear_behavior_policy_continuous,
            random_state=i,
        )
        # define evaluation policy using NNPolicyLearner
        nn_policy = ContinuousNNPolicyLearner(
            dim_context=dim_context,
            pg_method=pg_method,
            bandwidth=bandwidth,
            output_space=(min_action_value, max_action_value),
            hidden_layer_size=(10, 10),
            learning_rate_init=0.001,
            solver="sgd",
        )
        # baseline method 1. RandomPolicy
        random_policy = RandomPolicy(output_space=(min_action_value, max_action_value))
        # sample new training and test sets of synthetic logged bandit feedback
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(
            n_rounds=n_rounds,
        )
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(
            n_rounds=n_rounds,
        )
        # train the evaluation policy on the training set of the synthetic logged bandit feedback
        nn_policy.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
        )
        # predict the action decisions for the test set of the synthetic logged bandit feedback
        actions_predicted_by_nn_policy = nn_policy.predict(
            context=bandit_feedback_test["context"],
        )
        actions_predicted_by_random = random_policy.predict(
            context=bandit_feedback_test["context"],
        )
        # get the ground truth policy value for each learner
        gt_nn_policy_learner = dataset.calc_ground_truth_policy_value(
            context=bandit_feedback_test["context"],
            action=actions_predicted_by_nn_policy,
        )
        gt_random_policy = dataset.calc_ground_truth_policy_value(
            context=bandit_feedback_test["context"],
            action=actions_predicted_by_random,
        )

        return gt_nn_policy_learner, gt_random_policy

    n_runs = 10
    processed = Parallel(
        n_jobs=1,  # PyTorch uses multiple threads
        verbose=0,
    )([delayed(process)(i) for i in np.arange(n_runs)])
    list_gt_nn_policy, list_gt_random = [], []
    for i, ground_truth_policy_values in enumerate(processed):
        gt_nn_policy, gt_random = ground_truth_policy_values
        list_gt_nn_policy.append(gt_nn_policy)
        list_gt_random.append(gt_random)

    assert np.mean(list_gt_nn_policy) > np.mean(list_gt_random)
