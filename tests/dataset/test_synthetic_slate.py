from typing import List

import pytest
import numpy as np
import pandas as pd

from obp.dataset import (
    linear_reward_function,
    logistic_reward_function,
    linear_behavior_policy_logit,
    SyntheticSlateBanditDataset,
)

from obp.types import BanditFeedback

# n_unique_action, len_list, dim_context, reward_type, random_state, description
invalid_input_of_init = [
    ("4", 3, 2, "binary", 1, "n_unique_action must be an integer larger than 1"),
    (1, 3, 2, "binary", 1, "n_unique_action must be an integer larger than 1"),
    (5, "4", 2, "binary", 1, "len_list must be an integer such that"),
    (5, -1, 2, "binary", 1, "len_list must be an integer such that"),
    (5, 10, 2, "binary", 1, "len_list must be an integer such that"),
    (5, 3, 0, "binary", 1, "dim_context must be a positive integer"),
    (5, 3, "2", "binary", 1, "dim_context must be a positive integer"),
    (5, 3, 2, "aaa", 1, "reward_type must be either"),
    (5, 3, 2, "binary", "x", "random_state must be an integer"),
    (5, 3, 2, "binary", None, "random_state must be an integer"),
]


@pytest.mark.parametrize(
    "n_unique_action, len_list, dim_context, reward_type, random_state, description",
    invalid_input_of_init,
)
def test_synthetic_slate_init_using_invalid_inputs(
    n_unique_action, len_list, dim_context, reward_type, random_state, description
):
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = SyntheticSlateBanditDataset(
            n_unique_action=n_unique_action,
            len_list=len_list,
            dim_context=dim_context,
            reward_type=reward_type,
            random_state=random_state,
        )


def check_slate_bandit_feedback(bandit_feedback: BanditFeedback):
    # check pscore columns
    pscore_columns: List[str] = []
    pscore_candidate_columns = [
        "pscore_cascade",
        "pscore",
        "pscore_item_position",
    ]
    for column in pscore_candidate_columns:
        if column in bandit_feedback and bandit_feedback[column] is not None:
            pscore_columns.append(column)
        else:
            pscore_columns.append(column)
    assert (
        len(pscore_columns) > 0
    ), f"bandit feedback must contains at least one of the following pscore columns: {pscore_candidate_columns}"
    bandit_feedback_df = pd.DataFrame()
    for column in ["slate_id", "position", "action"] + pscore_columns:
        bandit_feedback_df[column] = bandit_feedback[column]
    # sort dataframe
    bandit_feedback_df = (
        bandit_feedback_df.sort_values(["slate_id", "position"])
        .reset_index(drop=True)
        .copy()
    )
    # check uniqueness
    assert (
        bandit_feedback_df.duplicated(["slate_id", "position"]).sum() == 0
    ), "position must not be duplicated in each impression"
    assert (
        bandit_feedback_df.duplicated(["slate_id", "action"]).sum() == 0
    ), "action must not be duplicated in each impression"
    # check pscores
    for column in pscore_columns:
        invalid_pscore_flgs = (bandit_feedback_df[column] < 0) | (
            bandit_feedback_df[column] > 1
        )
        assert invalid_pscore_flgs.sum() == 0, "the range of pscores must be [0, 1]"
    if "pscore_cascade" in pscore_columns and "pscore" in pscore_columns:
        assert (
            bandit_feedback_df["pscore_cascade"] < bandit_feedback_df["pscore"]
        ).sum() == 0, "pscore_cascade is smaller or equal to pscore"
    if "pscore_item_position" in pscore_columns and "pscore" in pscore_columns:
        assert (
            bandit_feedback_df["pscore_item_position"] < bandit_feedback_df["pscore"]
        ).sum() == 0, "pscore is smaller or equal to pscore_item_position"
    if "pscore_item_position" in pscore_columns and "pscore_cascade" in pscore_columns:
        assert (
            bandit_feedback_df["pscore_item_position"]
            < bandit_feedback_df["pscore_cascade"]
        ).sum() == 0, "pscore_cascade is smaller or equal to pscore_item_position"
    if "pscore_cascade" in pscore_columns:
        previous_minimum_pscore_cascade = (
            bandit_feedback_df.groupby("slate_id")["pscore_cascade"]
            .expanding()
            .min()
            .values
        )
        assert (
            previous_minimum_pscore_cascade < bandit_feedback_df["pscore_cascade"]
        ).sum() == 0, "pscore_cascade must be non-decresing sequence in each impression"
    if "pscore" in pscore_columns:
        count_pscore_in_expression = bandit_feedback_df.groupby("slate_id").apply(
            lambda x: x["pscore"].unique().shape[0]
        )
        assert (
            count_pscore_in_expression != 1
        ).sum() == 0, "pscore must be unique in each impression"
    if "pscore" in pscore_columns and "pscore_cascade" in pscore_columns:
        last_slot_feedback_df = bandit_feedback_df.drop_duplicates(
            "slate_id", keep="last"
        )
        assert (
            last_slot_feedback_df["pscore"] != last_slot_feedback_df["pscore_cascade"]
        ).sum() == 0, "pscore must be the same as pscore_cascade in the last slot"


def test_synthetic_slate_obtain_batch_bandit_feedback_using_uniform_random_behavior_policy():
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
    # get feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    pscore_columns = [
        "pscore_cascade",
        "pscore",
        "pscore_item_position",
    ]
    bandit_feedback_df = pd.DataFrame()
    for column in ["slate_id", "position", "action"] + pscore_columns:
        bandit_feedback_df[column] = bandit_feedback[column]
    # check pscore marginal
    pscore_item_position = float(len_list / n_unique_action)
    assert np.allclose(
        bandit_feedback_df["pscore_item_position"].unique(), [pscore_item_position]
    ), f"pscore_item_position must be [{pscore_item_position}], but {bandit_feedback_df['pscore_item_position'].unique()}"
    # check pscore joint
    pscore_cascade = []
    pscore_above = 1.0
    for position_ in np.arange(len_list):
        pscore_above = pscore_above * 1.0 / (n_unique_action - position_)
        pscore_cascade.append(pscore_above)
    assert np.allclose(
        bandit_feedback_df["pscore_cascade"], np.tile(pscore_cascade, n_rounds)
    ), f"pscore_cascade must be {pscore_cascade} for all impresessions"
    assert np.allclose(
        bandit_feedback_df["pscore"].unique(), [pscore_above]
    ), f"pscore must be {pscore_above} for all impressions"


def test_synthetic_slate_obtain_batch_bandit_feedback_using_uniform_random_behavior_policy_largescale():
    # set parameters
    n_unique_action = 100
    len_list = 10
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    n_rounds = 10000
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
    )
    # get feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_exact_uniform_pscore_item_position=True
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    # check pscore marginal
    pscore_item_position = float(len_list / n_unique_action)
    assert np.allclose(
        np.unique(bandit_feedback["pscore_item_position"]), [pscore_item_position]
    ), f"pscore_item_position must be [{pscore_item_position}], but {np.unique(bandit_feedback['pscore_item_position'])}"


def test_synthetic_slate_obtain_batch_bandit_feedback_using_linear_behavior_policy():
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
        behavior_policy_function=linear_behavior_policy_logit,
    )
    # get feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    # print reward
    pscore_columns = [
        "pscore_cascade",
        "pscore",
        "pscore_item_position",
    ]
    bandit_feedback_df = pd.DataFrame()
    for column in ["slate_id", "position", "action", "reward"] + pscore_columns:
        bandit_feedback_df[column] = bandit_feedback[column]
    print(bandit_feedback_df.groupby("position")["reward"].describe())
    if reward_type == "binary":
        assert set(np.unique(bandit_feedback["reward"])) == set([0, 1])


def test_synthetic_slate_obtain_batch_bandit_feedback_using_linear_behavior_policy_without_pscore_item_position():
    # set parameters
    n_unique_action = 80
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
        behavior_policy_function=linear_behavior_policy_logit,
    )
    # get feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_item_position=False
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    assert (
        bandit_feedback["pscore_item_position"] is None
    ), f"pscore marginal must be None, but {bandit_feedback['pscore_item_position']}"

    # random seed should be fixed
    dataset2 = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
        behavior_policy_function=linear_behavior_policy_logit,
    )
    # get feedback
    bandit_feedback2 = dataset2.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_item_position=False
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback2)
    # check random seed effect
    assert np.allclose(
        bandit_feedback["expected_reward_factual"],
        bandit_feedback2["expected_reward_factual"],
    )
    if reward_type == "binary":
        assert set(np.unique(bandit_feedback["reward"])) == set([0, 1])


# n_unique_action, len_list, dim_context, reward_type, random_state, n_rounds, reward_structure, click_model, behavior_policy_function, reward_function, return_pscore_item_position, description
valid_input_of_obtain_batch_bandit_feedback = [
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_additive",
        None,
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "standard_additive",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "independent",
        None,
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "independent",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "cascade_additive",
        None,
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "cascade_additive",
    ),
    (
        10,
        3,
        2,
        "continuous",
        123,
        1000,
        "standard_additive",
        None,
        linear_behavior_policy_logit,
        linear_reward_function,
        False,
        "standard_additive continuous",
    ),
    (
        10,
        3,
        2,
        "continuous",
        123,
        1000,
        "independent",
        None,
        linear_behavior_policy_logit,
        linear_reward_function,
        False,
        "independent continuous",
    ),
    (
        10,
        3,
        2,
        "continuous",
        123,
        1000,
        "cascade_additive",
        None,
        linear_behavior_policy_logit,
        linear_reward_function,
        False,
        "cascade_additive continuous",
    ),
    (
        10,
        3,
        2,
        "continuous",
        123,
        1000,
        "cascade_additive",
        None,
        None,
        None,
        False,
        "Random policy and reward function (continuous reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "cascade_exponential",
        None,
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "cascade_exponential (binary reward)",
    ),
    (
        10,
        3,
        2,
        "continuous",
        123,
        1000,
        "cascade_exponential",
        None,
        linear_behavior_policy_logit,
        linear_reward_function,
        False,
        "cascade_exponential (continuous reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_exponential",
        None,
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "standard_exponential (binary reward)",
    ),
    (
        10,
        3,
        2,
        "continuous",
        123,
        1000,
        "standard_exponential",
        None,
        linear_behavior_policy_logit,
        linear_reward_function,
        False,
        "standard_exponential (continuous reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "cascade_additive",
        "cascade",
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "cascade_additive, cascade click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "cascade_exponential",
        "cascade",
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "cascade_exponential, cascade click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_additive",
        "cascade",
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "standard_additive, cascade click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_exponential",
        "cascade",
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "standard_exponential, cascade click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "independent",
        "cascade",
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "independent, cascade click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "cascade_additive",
        "pbm",
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "cascade_additive, pbm click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "cascade_exponential",
        "pbm",
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "cascade_exponential, pbm click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_additive",
        "pbm",
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "standard_additive, pbm click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_exponential",
        "pbm",
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "standard_exponential, pbm click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "independent",
        "pbm",
        linear_behavior_policy_logit,
        logistic_reward_function,
        False,
        "independent, pbm click model (binary reward)",
    ),
]


@pytest.mark.parametrize(
    "n_unique_action, len_list, dim_context, reward_type, random_state, n_rounds, reward_structure, click_model, behavior_policy_function, reward_function, return_pscore_item_position, description",
    valid_input_of_obtain_batch_bandit_feedback,
)
def test_synthetic_slate_using_valid_inputs(
    n_unique_action,
    len_list,
    dim_context,
    reward_type,
    random_state,
    n_rounds,
    reward_structure,
    click_model,
    behavior_policy_function,
    reward_function,
    return_pscore_item_position,
    description,
):
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        random_state=random_state,
        behavior_policy_function=behavior_policy_function,
        base_reward_function=reward_function,
    )
    # get feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_item_position=return_pscore_item_position
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    pscore_columns = [
        "pscore_cascade",
        "pscore",
        "pscore_item_position",
    ]
    bandit_feedback_df = pd.DataFrame()
    for column in [
        "slate_id",
        "position",
        "action",
        "reward",
        "expected_reward_factual",
    ] + pscore_columns:
        bandit_feedback_df[column] = bandit_feedback[column]
    print(f"-------{description}--------")
    print(bandit_feedback_df.groupby("position")["reward"].describe())
    if reward_type == "binary":
        assert set(np.unique(bandit_feedback["reward"])) == set([0, 1])
