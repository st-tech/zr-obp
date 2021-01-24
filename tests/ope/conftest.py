from typing import Set, Tuple, List
from dataclasses import dataclass
import copy

import numpy as np
import pytest
from sklearn.utils import check_random_state

from obp.policy import Random, LogisticEpsilonGreedy
from obp.types import BanditFeedback
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
    linear_behavior_policy,
)
from obp.utils import sigmoid


@dataclass
class LogisticEpsilonGreedyBatch(LogisticEpsilonGreedy):
    """
    WIP: Add random action flag and compute_batch_action_dist method to LogisticEpsilonGreedy

    """

    def select_action(self, context: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Select action for new data.

        Parameters
        ----------
        context: array-like, shape (1, dim_context)
            Observed context vector.

        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.

        random action flag: bool
            Whether the action is randomly selected
        """
        if self.random_.rand() > self.epsilon:
            theta = np.array(
                [model.predict_proba(context) for model in self.model_list]
            ).flatten()
            return theta.argsort()[::-1][: self.len_list], False
        else:
            return (
                self.random_.choice(self.n_actions, size=self.len_list, replace=False),
                True,
            )

    def compute_batch_action_dist(
        self, context: np.ndarray
    ) -> Tuple[np.ndarray, List[bool]]:
        """Select action for new data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Observed context matrix.

        Returns
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Probability estimates of each arm being the best one for each sample, action, and position.

        """
        return np.array([1]), [False]


# generate synthetic dataset using SyntheticBanditDataset
@pytest.fixture(scope="session")
def synthetic_bandit_feedback() -> BanditFeedback:
    n_actions = 10
    dim_context = 5
    random_state = 12345
    n_rounds = 10000
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_function=logistic_reward_function,
        behavior_policy_function=linear_behavior_policy,
        random_state=random_state,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    return bandit_feedback


# make the expected reward of synthetic bandit feedback close to that of the Open Bandit Dataset
@pytest.fixture(scope="session")
def fixed_synthetic_bandit_feedback(synthetic_bandit_feedback) -> BanditFeedback:
    # set random
    random_state = 12345
    random_ = check_random_state(random_state)
    # copy synthetic bandit feedback
    bandit_feedback = copy.deepcopy(synthetic_bandit_feedback)
    # expected reward would be about 0.65%, which is close to that of ZOZO dataset
    logit = np.log(
        bandit_feedback["expected_reward"] / (1 - bandit_feedback["expected_reward"])
    )
    bandit_feedback["expected_reward"] = sigmoid(logit - 4.0)
    expected_reward_factual = bandit_feedback["expected_reward"][
        np.arange(bandit_feedback["n_rounds"]), bandit_feedback["action"]
    ]
    bandit_feedback["reward"] = random_.binomial(n=1, p=expected_reward_factual)
    return bandit_feedback


# key set of bandit feedback data
@pytest.fixture(scope="session")
def feedback_key_set() -> Set[str]:
    return {
        "action",
        "action_context",
        "context",
        "expected_reward",
        "n_actions",
        "n_rounds",
        "position",
        "pscore",
        "reward",
    }


# bandit_feedback["expected_reward"][0]
@pytest.fixture(scope="session")
def expected_reward_0() -> np.ndarray:
    return np.array(
        [
            0.80210203,
            0.73828559,
            0.83199558,
            0.83412285,
            0.7793723,
            0.50544696,
            0.7942911,
            0.81190503,
            0.70617705,
            0.68985306,
        ]
    )


# logistic evaluation policy
@pytest.fixture(scope="session")
def logistic_evaluation_policy(synthetic_bandit_feedback) -> LogisticEpsilonGreedy:
    random_state = 12345
    epsilon = 0.05
    dim = synthetic_bandit_feedback["context"].shape[1]
    n_actions = synthetic_bandit_feedback["n_actions"]
    evaluation_policy = LogisticEpsilonGreedy(
        dim=dim,
        n_actions=n_actions,
        len_list=synthetic_bandit_feedback["position"].ndim,
        random_state=random_state,
        epsilon=epsilon,
    )
    # set coef_ of evaluation policy
    random_ = check_random_state(random_state)
    for action in range(n_actions):
        evaluation_policy.model_list[action]._m = random_.uniform(size=dim)
    return evaluation_policy


# logistic evaluation policy
@pytest.fixture(scope="session")
def logistic_batch_action_dist(logistic_evaluation_policy) -> np.ndarray:
    return np.array([1])


# random evaluation policy
@pytest.fixture(scope="session")
def random_action_dist(synthetic_bandit_feedback) -> np.ndarray:
    n_actions = synthetic_bandit_feedback["n_actions"]
    evaluation_policy = Random(
        n_actions=n_actions, len_list=synthetic_bandit_feedback["position"].ndim
    )
    action_dist = evaluation_policy.compute_batch_action_dist(
        n_rounds=synthetic_bandit_feedback["n_rounds"]
    )
    return action_dist
