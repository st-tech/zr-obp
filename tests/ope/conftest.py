import copy
import os
from typing import Set

import numpy as np
import pytest
from scipy import special
from sklearn.utils import check_random_state

from obp.dataset import linear_behavior_policy
from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticBanditDataset
from obp.dataset import SyntheticBanditDatasetWithActionEmbeds
from obp.dataset import SyntheticContinuousBanditDataset
from obp.dataset import SyntheticMultiLoggersBanditDataset
from obp.dataset import SyntheticSlateBanditDataset
from obp.policy import Random
from obp.types import BanditFeedback
from obp.utils import sigmoid


# resolve ImportMismatchError when using virtual environment
os.environ["PY_IGNORE_IMPORTMISMATCH"] = "1"


# generate synthetic bandit dataset using SyntheticBanditDataset
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


# generate synthetic slate bandit dataset using SyntheticSlateBanditDataset
@pytest.fixture(scope="session")
def synthetic_slate_bandit_feedback() -> BanditFeedback:
    # set parameters
    n_unique_action = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 100
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    return bandit_feedback


# generate synthetic continuous bandit dataset using SyntheticContinuousBanditDataset
@pytest.fixture(scope="session")
def synthetic_continuous_bandit_feedback() -> BanditFeedback:
    # set parameters
    dim_context = 2
    random_state = 12345
    n_rounds = 100
    min_action_value = -10
    max_action_value = 10
    dataset = SyntheticContinuousBanditDataset(
        dim_context=dim_context,
        min_action_value=min_action_value,
        max_action_value=max_action_value,
        random_state=random_state,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    return bandit_feedback


@pytest.fixture(scope="session")
def synthetic_multi_bandit_feedback() -> BanditFeedback:
    n_actions = 10
    dim_context = 5
    betas = [-10, -5, 0, 5, 10]
    rhos = [1, 2, 3, 2, 1]
    random_state = 12345
    n_rounds = 10000
    dataset = SyntheticMultiLoggersBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        betas=betas,
        rhos=rhos,
        reward_function=logistic_reward_function,
        random_state=random_state,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    return bandit_feedback


@pytest.fixture(scope="session")
def synthetic_bandit_feedback_with_embed() -> BanditFeedback:
    n_actions = 10
    dim_context = 5
    n_cat_dim = 3
    n_cat_per_dim = 5
    random_state = 12345
    n_rounds = 10000
    dataset = SyntheticBanditDatasetWithActionEmbeds(
        n_actions=n_actions,
        dim_context=dim_context,
        n_cat_dim=n_cat_dim,
        n_cat_per_dim=n_cat_per_dim,
        reward_function=logistic_reward_function,
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
    # expected reward would be about 0.65%, which is close to that of the ZOZO dataset
    logit = special.logit(bandit_feedback["expected_reward"])
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
        "pi_b",
        "pscore",
        "reward",
    }


# random evaluation policy
@pytest.fixture(scope="session")
def random_action_dist(synthetic_bandit_feedback) -> np.ndarray:
    n_actions = synthetic_bandit_feedback["n_actions"]
    evaluation_policy = Random(n_actions=n_actions, len_list=1)
    action_dist = evaluation_policy.compute_batch_action_dist(
        n_rounds=synthetic_bandit_feedback["n_rounds"]
    )
    return action_dist


def generate_action_dist(i, j, k):
    x = np.random.uniform(size=(i, j, k))
    action_dist = x / x.sum(axis=1)[:, np.newaxis, :]
    return action_dist
