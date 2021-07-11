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

# n_unique_action, len_list, dim_context, reward_type, reward_structure, decay_function, click_model, eta, random_state, err, description
invalid_input_of_init = [
    (
        "4",
        3,
        2,
        "binary",
        "independent",
        "exponential",
        "pbm",
        1.0,
        1,
        ValueError,
        "n_unique_action must be an integer larger than 1",
    ),
    (
        1,
        3,
        2,
        "binary",
        "independent",
        "exponential",
        "pbm",
        1.0,
        1,
        ValueError,
        "n_unique_action must be an integer larger than 1",
    ),
    (
        5,
        "4",
        2,
        "binary",
        "independent",
        "exponential",
        "pbm",
        1.0,
        1,
        ValueError,
        "len_list must be an integer larger than",
    ),
    (
        5,
        -1,
        2,
        "binary",
        "independent",
        "exponential",
        "pbm",
        1.0,
        1,
        ValueError,
        "len_list must be an integer larger than",
    ),
    (
        5,
        10,
        2,
        "binary",
        "independent",
        "exponential",
        "pbm",
        1.0,
        1,
        ValueError,
        "len_list must be equal to or smaller than",
    ),
    (
        5,
        3,
        0,
        "binary",
        "independent",
        "exponential",
        "pbm",
        1.0,
        1,
        ValueError,
        "dim_context must be a positive integer",
    ),
    (
        5,
        3,
        "2",
        "binary",
        "independent",
        "exponential",
        "pbm",
        1.0,
        1,
        ValueError,
        "dim_context must be a positive integer",
    ),
    (
        5,
        3,
        2,
        "aaa",
        "independent",
        "exponential",
        "pbm",
        1.0,
        1,
        ValueError,
        "reward_type must be either",
    ),
    (
        5,
        3,
        2,
        "binary",
        "aaa",
        "exponential",
        "pbm",
        1.0,
        1,
        ValueError,
        "reward_structure must be one of",
    ),
    (
        5,
        3,
        2,
        "binary",
        "independent",
        "aaa",
        "pbm",
        1.0,
        1,
        ValueError,
        "decay_function must be either",
    ),
    (
        5,
        3,
        2,
        "binary",
        "independent",
        "exponential",
        "aaa",
        1.0,
        1,
        ValueError,
        "click_model must be one of",
    ),
    (
        5,
        3,
        2,
        "binary",
        "independent",
        "exponential",
        "pbm",
        "aaa",
        1,
        TypeError,
        "`eta` must be an instance of <class 'float'>, not <class 'str'>.",
    ),
    (
        5,
        3,
        2,
        "binary",
        "independent",
        "exponential",
        "pbm",
        -1.0,
        1,
        ValueError,
        "`eta`= -1.0, must be >= 0.0.",
    ),
    (
        5,
        3,
        2,
        "binary",
        "independent",
        "exponential",
        "pbm",
        1.0,
        "x",
        ValueError,
        "random_state must be an integer",
    ),
    (
        5,
        3,
        2,
        "binary",
        "independent",
        "exponential",
        "pbm",
        1.0,
        None,
        ValueError,
        "random_state must be an integer",
    ),
]


@pytest.mark.parametrize(
    "n_unique_action, len_list, dim_context, reward_type, reward_structure, decay_function, click_model, eta, random_state, err, description",
    invalid_input_of_init,
)
def test_synthetic_slate_init_using_invalid_inputs(
    n_unique_action,
    len_list,
    dim_context,
    reward_type,
    reward_structure,
    decay_function,
    click_model,
    eta,
    random_state,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = SyntheticSlateBanditDataset(
            n_unique_action=n_unique_action,
            len_list=len_list,
            dim_context=dim_context,
            reward_type=reward_type,
            reward_structure=reward_structure,
            decay_function=decay_function,
            click_model=click_model,
            eta=eta,
            random_state=random_state,
        )


def check_slate_bandit_feedback(
    bandit_feedback: BanditFeedback, is_factorizable: bool = False
):
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
    ), f"bandit feedback must contain at least one of the following pscore columns: {pscore_candidate_columns}"
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
    ), "position must not be duplicated in each slate"
    assert (
        bandit_feedback_df.duplicated(["slate_id", "action"]).sum() == 0
        if not is_factorizable
        else True
    ), "action must not be duplicated in each slate"
    # check pscores
    for column in pscore_columns:
        invalid_pscore_flgs = (bandit_feedback_df[column] < 0) | (
            bandit_feedback_df[column] > 1
        )
        assert invalid_pscore_flgs.sum() == 0, "the range of pscores must be [0, 1]"
    if "pscore_cascade" in pscore_columns and "pscore" in pscore_columns:
        assert (
            bandit_feedback_df["pscore_cascade"] < bandit_feedback_df["pscore"]
        ).sum() == 0, "pscore must be smaller than or equal to pscore_cascade"
    if "pscore_item_position" in pscore_columns and "pscore" in pscore_columns:
        assert (
            bandit_feedback_df["pscore_item_position"] < bandit_feedback_df["pscore"]
        ).sum() == 0, "pscore must be smaller than or equal to pscore_item_position"
    if "pscore_item_position" in pscore_columns and "pscore_cascade" in pscore_columns:
        assert (
            bandit_feedback_df["pscore_item_position"]
            < bandit_feedback_df["pscore_cascade"]
        ).sum() == 0, (
            "pscore_cascade must be smaller than or equal to pscore_item_position"
        )
    if "pscore_cascade" in pscore_columns:
        previous_minimum_pscore_cascade = (
            bandit_feedback_df.groupby("slate_id")["pscore_cascade"]
            .expanding()
            .min()
            .values
        )
        assert (
            previous_minimum_pscore_cascade < bandit_feedback_df["pscore_cascade"]
        ).sum() == 0, "pscore_cascade must be non-decresing sequence in each slate"
    if "pscore" in pscore_columns:
        count_pscore_in_expression = bandit_feedback_df.groupby("slate_id").apply(
            lambda x: x["pscore"].unique().shape[0]
        )
        assert (
            count_pscore_in_expression != 1
        ).sum() == 0, "pscore must be unique in each slate"
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
    # obtain feedback
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
    pscore_item_position = 1 / n_unique_action
    assert np.allclose(
        bandit_feedback_df["pscore_item_position"].unique(), pscore_item_position
    ), f"pscore_item_position must be [{pscore_item_position}], but {bandit_feedback_df['pscore_item_position'].unique()}"
    # check pscore joint
    pscore_cascade = []
    pscore_above = 1.0
    for position_ in np.arange(len_list):
        pscore_above *= 1.0 / (n_unique_action - position_)
        pscore_cascade.append(pscore_above)
    assert np.allclose(
        bandit_feedback_df["pscore_cascade"], np.tile(pscore_cascade, n_rounds)
    ), f"pscore_cascade must be {pscore_cascade} for all slates"
    assert np.allclose(
        bandit_feedback_df["pscore"].unique(), [pscore_above]
    ), f"pscore must be {pscore_above} for all slates"


def test_synthetic_slate_obtain_batch_bandit_feedback_using_uniform_random_factorizable_behavior_policy():
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
        is_factorizable=True,
        random_state=random_state,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback, is_factorizable=True)
    pscore_columns = [
        "pscore_cascade",
        "pscore",
        "pscore_item_position",
    ]
    bandit_feedback_df = pd.DataFrame()
    for column in ["slate_id", "position", "action"] + pscore_columns:
        bandit_feedback_df[column] = bandit_feedback[column]
    # check pscore marginal
    pscore_item_position = 1 / n_unique_action
    assert np.allclose(
        bandit_feedback_df["pscore_item_position"].unique(), pscore_item_position
    ), f"pscore_item_position must be [{pscore_item_position}], but {bandit_feedback_df['pscore_item_position'].unique()}"
    # check pscore joint
    pscore_cascade = []
    pscore_above = 1.0
    for position_ in np.arange(len_list):
        pscore_above *= 1.0 / n_unique_action
        pscore_cascade.append(pscore_above)
    assert np.allclose(
        bandit_feedback_df["pscore_cascade"], np.tile(pscore_cascade, n_rounds)
    ), f"pscore_cascade must be {pscore_cascade} for all slates"
    assert np.allclose(
        bandit_feedback_df["pscore"].unique(), [pscore_above]
    ), f"pscore must be {pscore_above} for all slates"


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
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(bandit_feedback=bandit_feedback)
    # check pscore marginal
    pscore_item_position = 1 / n_unique_action
    assert np.allclose(
        np.unique(bandit_feedback["pscore_item_position"]), pscore_item_position
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
    with pytest.raises(ValueError):
        _ = dataset.obtain_batch_bandit_feedback(n_rounds=-1)
    with pytest.raises(ValueError):
        _ = dataset.obtain_batch_bandit_feedback(n_rounds="a")

    # obtain feedback
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
    # obtain feedback
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
    # obtain feedback
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


# n_unique_action, len_list, dim_context, reward_type, decay_function, random_state, n_rounds, reward_structure, click_model, eta, behavior_policy_function, is_factorizable, reward_function, return_pscore_item_position, description
valid_input_of_obtain_batch_bandit_feedback = [
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_additive",
        "exponential",
        None,
        1.0,
        linear_behavior_policy_logit,
        False,
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
        "exponential",
        None,
        1.0,
        linear_behavior_policy_logit,
        False,
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
        "exponential",
        None,
        1.0,
        linear_behavior_policy_logit,
        False,
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
        "exponential",
        None,
        1.0,
        linear_behavior_policy_logit,
        False,
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
        "exponential",
        None,
        1.0,
        linear_behavior_policy_logit,
        False,
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
        "exponential",
        None,
        1.0,
        linear_behavior_policy_logit,
        False,
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
        "exponential",
        None,
        0.0,
        None,
        False,
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
        "cascade_decay",
        "exponential",
        None,
        0.0,
        linear_behavior_policy_logit,
        False,
        logistic_reward_function,
        False,
        "cascade_decay (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "cascade_decay",
        "inverse",
        None,
        0.0,
        linear_behavior_policy_logit,
        False,
        logistic_reward_function,
        False,
        "cascade_decay (binary reward)",
    ),
    (
        10,
        3,
        2,
        "continuous",
        123,
        1000,
        "cascade_decay",
        "exponential",
        None,
        0.0,
        linear_behavior_policy_logit,
        False,
        linear_reward_function,
        False,
        "cascade_decay (continuous reward)",
    ),
    (
        10,
        3,
        2,
        "continuous",
        123,
        1000,
        "cascade_decay",
        "inverse",
        None,
        0.0,
        linear_behavior_policy_logit,
        False,
        linear_reward_function,
        False,
        "cascade_decay (continuous reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_decay",
        "exponential",
        None,
        0.0,
        linear_behavior_policy_logit,
        False,
        logistic_reward_function,
        False,
        "standard_decay (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_decay",
        "inverse",
        None,
        0.0,
        linear_behavior_policy_logit,
        False,
        logistic_reward_function,
        False,
        "standard_decay (binary reward)",
    ),
    (
        10,
        3,
        2,
        "continuous",
        123,
        1000,
        "standard_decay",
        "exponential",
        None,
        0.0,
        linear_behavior_policy_logit,
        False,
        linear_reward_function,
        False,
        "standard_decay (continuous reward)",
    ),
    (
        10,
        3,
        2,
        "continuous",
        123,
        1000,
        "standard_decay",
        "inverse",
        None,
        0.0,
        linear_behavior_policy_logit,
        False,
        linear_reward_function,
        False,
        "standard_decay (continuous reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "cascade_additive",
        "exponential",
        "cascade",
        0.0,
        linear_behavior_policy_logit,
        False,
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
        "cascade_decay",
        "exponential",
        "cascade",
        0.5,
        linear_behavior_policy_logit,
        False,
        logistic_reward_function,
        False,
        "cascade_decay, cascade click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_additive",
        "exponential",
        "cascade",
        0.5,
        linear_behavior_policy_logit,
        False,
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
        "standard_decay",
        "exponential",
        "cascade",
        0.5,
        linear_behavior_policy_logit,
        False,
        logistic_reward_function,
        False,
        "standard_decay, cascade click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "independent",
        "exponential",
        "cascade",
        0.5,
        linear_behavior_policy_logit,
        False,
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
        "exponential",
        "pbm",
        0.5,
        linear_behavior_policy_logit,
        False,
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
        "cascade_decay",
        "exponential",
        "pbm",
        0.5,
        linear_behavior_policy_logit,
        False,
        logistic_reward_function,
        False,
        "cascade_decay, pbm click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "standard_additive",
        "exponential",
        "pbm",
        0.5,
        linear_behavior_policy_logit,
        False,
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
        "standard_decay",
        "exponential",
        "pbm",
        0.5,
        linear_behavior_policy_logit,
        False,
        logistic_reward_function,
        False,
        "standard_decay, pbm click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "independent",
        "exponential",
        "pbm",
        0.5,
        linear_behavior_policy_logit,
        False,
        logistic_reward_function,
        False,
        "independent, pbm click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "independent",
        "exponential",
        "pbm",
        0.5,
        linear_behavior_policy_logit,
        True,
        logistic_reward_function,
        False,
        "independent, pbm click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "independent",
        "exponential",
        "pbm",
        0.5,
        None,
        False,
        logistic_reward_function,
        False,
        "independent, pbm click model (binary reward)",
    ),
    (
        10,
        3,
        2,
        "binary",
        123,
        1000,
        "independent",
        "exponential",
        "pbm",
        0.5,
        None,
        True,
        logistic_reward_function,
        False,
        "independent, pbm click model (binary reward)",
    ),
    (
        3,
        5,
        2,
        "binary",
        123,
        1000,
        "independent",
        "exponential",
        "pbm",
        0.5,
        None,
        True,
        logistic_reward_function,
        False,
        "independent, pbm click model (binary reward)",
    ),
]


@pytest.mark.parametrize(
    "n_unique_action, len_list, dim_context, reward_type, random_state, n_rounds, reward_structure, decay_function, click_model, eta, behavior_policy_function, is_factorizable, reward_function, return_pscore_item_position, description",
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
    decay_function,
    click_model,
    eta,
    behavior_policy_function,
    is_factorizable,
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
        decay_function=decay_function,
        click_model=click_model,
        eta=eta,
        random_state=random_state,
        behavior_policy_function=behavior_policy_function,
        is_factorizable=is_factorizable,
        base_reward_function=reward_function,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_item_position=return_pscore_item_position
    )
    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(
        bandit_feedback=bandit_feedback, is_factorizable=is_factorizable
    )
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


n_rounds = 5
len_list = 3
# slate_id, reward, description
invalid_input_of_calc_on_policy_policy_value = [
    (
        np.repeat(np.arange(n_rounds), len_list),
        "4",  #
        "reward must be ndarray",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros((n_rounds, len_list), dtype=int),  #
        "reward must be 1-dimensional",
    ),
    (
        "4",  #
        np.zeros(n_rounds * len_list, dtype=int),
        "slate_id must be ndarray",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list).reshape((n_rounds, len_list)),  #
        np.zeros(n_rounds * len_list, dtype=int),
        "slate_id must be 1-dimensional",
    ),
    (
        np.repeat(np.arange(n_rounds), len_list),
        np.zeros(n_rounds * len_list - 1, dtype=int),  #
        "the size of axis 0 of reward must be the same as that of slate_id",
    ),
]


@pytest.mark.parametrize(
    "slate_id, reward, description",
    invalid_input_of_calc_on_policy_policy_value,
)
def test_calc_on_policy_policy_value_using_invalid_input_data(
    slate_id, reward, description
) -> None:
    # set parameters
    n_unique_action = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
    )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = dataset.calc_on_policy_policy_value(reward=reward, slate_id=slate_id)


# slate_id, reward, description
valid_input_of_calc_on_policy_policy_value = [
    (
        np.array([1, 1, 2, 2, 3, 4]),
        np.array([0, 1, 1, 0, 0, 0]),
        0.5,
        "4 slate ids",
    ),
    (
        np.array([1, 1]),
        np.array([2, 3]),
        5,
        "one slate id",
    ),
]


@pytest.mark.parametrize(
    "slate_id, reward, result, description",
    valid_input_of_calc_on_policy_policy_value,
)
def test_calc_on_policy_policy_value_using_valid_input_data(
    slate_id, reward, result, description
) -> None:
    # set parameters
    n_unique_action = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
        behavior_policy_function=linear_behavior_policy_logit,
    )
    assert result == dataset.calc_on_policy_policy_value(
        reward=reward, slate_id=slate_id
    )


# evaluation_policy_type, epsilon, context, action, err, description
invalid_input_of_generate_evaluation_policy_pscore = [
    (
        "awesome",  #
        1.0,
        np.ones([5, 2]),
        np.tile(np.arange(3), 5),
        ValueError,
        "evaluation_policy_type must be",
    ),
    (
        "optimal",
        1.0,
        np.array([5, 2]),  #
        np.tile(np.arange(3), 5),
        ValueError,
        "context must be 2-dimensional ndarray",
    ),
    (
        "optimal",
        1.0,
        np.ones([5, 2]),
        np.ones([5, 2]),  #
        ValueError,
        "action must be 1-dimensional ndarray",
    ),
    (
        "optimal",
        1.0,
        np.ones([5, 2]),
        np.random.choice(5),  #
        ValueError,
        "action must be 1-dimensional ndarray",
    ),
    (
        "optimal",
        1.0,
        np.ones([5, 2]),
        np.ones(5),  #
        ValueError,
        "action must be 1-dimensional ndarray, shape (n_rounds * len_list)",
    ),
    (
        "optimal",
        "aaa",  #
        np.ones([5, 2]),
        np.tile(np.arange(3), 5),
        TypeError,
        "`epsilon` must be an instance of <class 'float'>, not <class 'str'>.",
    ),
    (
        "optimal",
        -1.0,  #
        np.ones([5, 2]),
        np.tile(np.arange(3), 5),
        ValueError,
        "`epsilon`= -1.0, must be >= 0.0.",
    ),
    (
        "optimal",
        2.0,  #
        np.ones([5, 2]),
        np.tile(np.arange(3), 5),
        ValueError,
        "`epsilon`= 2.0, must be <= 1.0.",
    ),
]


@pytest.mark.parametrize(
    "evaluation_policy_type, epsilon, context, action, err, description",
    invalid_input_of_generate_evaluation_policy_pscore,
)
def test_generate_evaluation_policy_pscore_using_invalid_input_data(
    evaluation_policy_type,
    epsilon,
    context,
    action,
    err,
    description,
) -> None:
    # set parameters
    n_unique_action = 10
    len_list = 3
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
        base_reward_function=logistic_reward_function,
    )
    with pytest.raises(err, match=f"{description}*"):
        _ = dataset.generate_evaluation_policy_pscore(
            evaluation_policy_type=evaluation_policy_type,
            epsilon=epsilon,
            context=context,
            action=action,
        )


# n_unique_action, is_factorizable, evaluation_policy_type, epsilon, description
valid_input_of_generate_evaluation_policy_pscore = [
    (
        10,
        False,
        "optimal",
        0.1,
        "optimal evaluation policy",
    ),
    (
        10,
        True,
        "optimal",
        0.1,
        "optimal evaluation policy",
    ),
    (
        10,
        False,
        "anti-optimal",
        0.1,
        "anti-optimal evaluation policy",
    ),
    (
        10,
        True,
        "random",
        None,
        "random evaluation policy",
    ),
    (
        10,
        False,
        "optimal",
        0.0,
        "optimal evaluation policy, epsilon=0.0 (greedy)",
    ),
    (
        10,
        True,
        "optimal",
        1.0,
        "optimal evaluation policy, epsilon=1.0 (random)",
    ),
    (
        2,
        True,
        "optimal",
        1.0,
        "optimal evaluation policy, epsilon=1.0 (random)",
    ),
]


@pytest.mark.parametrize(
    "n_unique_action, is_factorizable, evaluation_policy_type, epsilon, description",
    valid_input_of_generate_evaluation_policy_pscore,
)
def test_generate_evaluation_policy_pscore_using_valid_input_data(
    n_unique_action,
    is_factorizable,
    evaluation_policy_type,
    epsilon,
    description,
) -> None:
    # set parameters
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
        is_factorizable=is_factorizable,
        base_reward_function=logistic_reward_function,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds, return_pscore_item_position=True
    )
    # generate pscores
    (
        pscore,
        pscore_item_position,
        pscore_cascade,
    ) = dataset.generate_evaluation_policy_pscore(
        evaluation_policy_type=evaluation_policy_type,
        context=bandit_feedback["context"],
        epsilon=epsilon,
        action=bandit_feedback["action"],
    )
    if evaluation_policy_type == "random" or epsilon == 1.0:
        # pscores of random evaluation policy must be the same as those of bandit feedback using random behavior policy
        assert np.allclose(pscore, bandit_feedback["pscore"])
        assert np.allclose(
            pscore_item_position, bandit_feedback["pscore_item_position"]
        )
        assert np.allclose(pscore_cascade, bandit_feedback["pscore_cascade"])
    if epsilon == 0.0:
        # pscore element of greedy evaluation policy must be either 0 or 1
        assert len(set(np.unique(pscore)) - set([0.0, 1.0])) == 0
        assert len(set(np.unique(pscore_item_position)) - set([0.0, 1.0])) == 0
        assert len(set(np.unique(pscore_cascade)) - set([0.0, 1.0])) == 0
    # check pscores
    assert (
        pscore_cascade < pscore
    ).sum() == 0, "pscore must be smaller than or equal to pscore_cascade"
    assert (
        pscore_item_position < pscore
    ).sum() == 0, "pscore must be smaller than or equal to pscore_item_position"
    assert (
        pscore_item_position < pscore_cascade
    ).sum() == 0, "pscore_cascade must be smaller than or equal to pscore_item_position"

    # check slate bandit feedback (common test)
    check_slate_bandit_feedback(
        bandit_feedback=bandit_feedback, is_factorizable=is_factorizable
    )
    bandit_feedback_df = pd.DataFrame()
    for column in ["slate_id", "position", "action"]:
        bandit_feedback_df[column] = bandit_feedback[column]
    bandit_feedback_df["pscore"] = pscore
    bandit_feedback_df["pscore_cascade"] = pscore_cascade
    bandit_feedback_df["pscore_item_position"] = pscore_item_position

    previous_minimum_pscore_cascade = (
        bandit_feedback_df.groupby("slate_id")["pscore_cascade"]
        .expanding()
        .min()
        .values
    )
    assert (
        previous_minimum_pscore_cascade < pscore_cascade
    ).sum() == 0, "pscore_cascade must be non-decresing sequence in each slate"
    count_pscore_in_expression = bandit_feedback_df.groupby("slate_id").apply(
        lambda x: x["pscore"].unique().shape[0]
    )
    assert (
        count_pscore_in_expression != 1
    ).sum() == 0, "pscore must be unique in each slate"
    last_slot_feedback_df = bandit_feedback_df.drop_duplicates("slate_id", keep="last")
    assert np.allclose(
        last_slot_feedback_df["pscore"], last_slot_feedback_df["pscore_cascade"]
    ), "pscore must be the same as pscore_cascade in the last slot"


# n_unique_action, len_list, epsilon, action_2d, sorted_actions, random_pscore, random_pscore_item_position, random_pscore_cascade, true_pscore, true_pscore_item_position, true_pscore_cascade, description
valid_input_of_calc_epsilon_greedy_pscore = [
    (
        5,
        3,
        0.1,
        np.tile(np.arange(3), 4).reshape((4, 3)),
        np.array([[0, 1, 2], [0, 1, 3], [1, 0, 2], [1, 0, 4]]),
        np.ones(12) / 60,  # 1 / 5P3
        np.ones(12) / 5,  # 1/ 5
        np.tile([1 / 5, 1 / 20, 1 / 60], 4),
        np.array(
            [[0.9 + 0.1 / 60] * 3, [0.1 / 60] * 3, [0.1 / 60] * 3, [0.1 / 60] * 3]
        ).flatten(),
        np.array(
            [
                [0.9 + 0.1 / 5] * 3,
                [0.9 + 0.1 / 5, 0.9 + 0.1 / 5, 0.1 / 5],
                [0.1 / 5, 0.1 / 5, 0.9 + 0.1 / 5],
                [0.1 / 5] * 3,
            ]
        ).flatten(),
        np.array(
            [
                [0.9 + 0.1 / 5, 0.9 + 0.1 / 20, 0.9 + 0.1 / 60],
                [0.9 + 0.1 / 5, 0.9 + 0.1 / 20, 0.1 / 60],
                [0.1 / 5, 0.1 / 20, 0.1 / 60],
                [0.1 / 5, 0.1 / 20, 0.1 / 60],
            ]
        ).flatten(),
        "epsilon is 0.1",
    ),
]


@pytest.mark.parametrize(
    "n_unique_action, len_list, epsilon, action_2d, sorted_actions, random_pscore, random_pscore_item_position, random_pscore_cascade, true_pscore, true_pscore_item_position, true_pscore_cascade, description",
    valid_input_of_calc_epsilon_greedy_pscore,
)
def test_calc_epsilon_greedy_pscore_using_valid_input_data(
    n_unique_action,
    len_list,
    epsilon,
    action_2d,
    sorted_actions,
    random_pscore,
    random_pscore_item_position,
    random_pscore_cascade,
    true_pscore,
    true_pscore_item_position,
    true_pscore_cascade,
    description,
) -> None:
    # set parameters
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
        base_reward_function=logistic_reward_function,
    )
    (
        pscore,
        pscore_item_position,
        pscore_cascade,
    ) = dataset._calc_epsilon_greedy_pscore(
        epsilon=epsilon,
        action_2d=action_2d,
        sorted_actions=sorted_actions,
        random_pscore=random_pscore,
        random_pscore_item_position=random_pscore_item_position,
        random_pscore_cascade=random_pscore_cascade,
    )
    assert np.allclose(true_pscore, pscore)
    assert np.allclose(true_pscore_item_position, pscore_item_position)
    assert np.allclose(true_pscore_cascade, pscore_cascade)


# n_rounds, n_unique_action, len_list, dim_context, reward_type, reward_structure, click_model, evaluation_policy_logit_, context, err, description
invalid_input_of_calc_ground_truth_policy_value = [
    (
        3,
        3,
        2,
        2,
        "binary",
        "independent",
        None,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]).flatten(),
        np.ones((3, 2)),
        ValueError,
        "evaluation_policy_logit_ must be 2-dimensional",
    ),
    (
        3,
        2,
        2,
        2,
        "binary",
        "independent",
        None,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        np.ones((3, 2)),
        ValueError,
        "the size of axis 1 of evaluation_policy_logit_ must be",
    ),
    (
        3,
        3,
        2,
        1,
        "binary",
        "independent",
        None,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        np.ones((3, 2)),
        ValueError,
        "the size of axis 1 of context must be",
    ),
    (
        4,
        3,
        2,
        2,
        "binary",
        "independent",
        None,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        np.ones((3, 2)),
        ValueError,
        "the length of evaluation_policy_logit_ and context",
    ),
    (
        3,
        3,
        2,
        2,
        "binary",
        "independent",
        None,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        np.ones((3, 2)),
        ValueError,
        "the length of evaluation_policy_logit_ and context",
    ),
]


@pytest.mark.parametrize(
    "n_rounds, n_unique_action, len_list, dim_context, reward_type, reward_structure, click_model, evaluation_policy_logit_, context, err, description",
    invalid_input_of_calc_ground_truth_policy_value,
)
def test_calc_ground_truth_policy_value_using_invalid_input_data(
    n_rounds,
    n_unique_action,
    len_list,
    dim_context,
    reward_type,
    reward_structure,
    click_model,
    evaluation_policy_logit_,
    context,
    err,
    description,
):
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        base_reward_function=logistic_reward_function,
    )
    _ = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    with pytest.raises(err, match=f"{description}*"):
        dataset.calc_ground_truth_policy_value(
            evaluation_policy_logit_=evaluation_policy_logit_,
            context=context,
        )


# n_rounds, n_unique_action, len_list, dim_context, reward_type, reward_structure, click_model, base_reward_function, is_factorizable, evaluation_policy_logit_, description
valid_input_of_calc_ground_truth_policy_value = [
    (
        4,
        3,
        2,
        2,
        "binary",
        "independent",
        None,
        logistic_reward_function,
        False,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
    (
        3,
        2,
        2,
        1,
        "binary",
        "independent",
        None,
        logistic_reward_function,
        False,
        np.array([[1, 2], [3, 4], [5, 6]]),
        None,
    ),
    (
        4,
        3,
        2,
        2,
        "binary",
        "independent",
        None,
        None,
        False,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
    (
        4,
        3,
        2,
        2,
        "binary",
        "cascade_decay",
        None,
        logistic_reward_function,
        False,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
    (
        4,
        3,
        2,
        2,
        "binary",
        "cascade_additive",
        None,
        logistic_reward_function,
        False,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
    (
        4,
        3,
        2,
        2,
        "binary",
        "standard_decay",
        None,
        logistic_reward_function,
        False,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
    (
        4,
        3,
        2,
        2,
        "binary",
        "standard_additive",
        None,
        logistic_reward_function,
        False,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
    (
        4,
        3,
        2,
        2,
        "continuous",
        "cascade_decay",
        None,
        logistic_reward_function,
        False,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
    (
        4,
        3,
        2,
        2,
        "binary",
        "cascade_decay",
        "pbm",
        logistic_reward_function,
        False,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
    (
        4,
        3,
        2,
        2,
        "binary",
        "cascade_decay",
        "cascade",
        logistic_reward_function,
        False,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
    (
        4,
        3,
        2,
        2,
        "binary",
        "cascade_decay",
        "cascade",
        logistic_reward_function,
        True,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
    (
        4,
        3,
        5,
        2,
        "binary",
        "cascade_decay",
        "cascade",
        logistic_reward_function,
        True,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]),
        None,
    ),
]


@pytest.mark.parametrize(
    "n_rounds, n_unique_action, len_list, dim_context, reward_type, reward_structure, click_model, base_reward_function, is_factorizable, evaluation_policy_logit_, description",
    valid_input_of_calc_ground_truth_policy_value,
)
def test_calc_ground_truth_policy_value_using_valid_input_data(
    n_rounds,
    n_unique_action,
    len_list,
    dim_context,
    reward_type,
    reward_structure,
    click_model,
    base_reward_function,
    is_factorizable,
    evaluation_policy_logit_,
    description,
):
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        base_reward_function=base_reward_function,
        is_factorizable=is_factorizable,
    )
    logged_bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    policy_value = dataset.calc_ground_truth_policy_value(
        evaluation_policy_logit_=evaluation_policy_logit_,
        context=logged_bandit_feedback["context"],
    )
    assert isinstance(policy_value, float) and 0 <= policy_value


@pytest.mark.parametrize("is_factorizable", [(True), (False)])
def test_calc_ground_truth_policy_value_value_check_with_click_model(is_factorizable):
    n_rounds = 3
    n_unique_action = 4
    len_list = 3
    dim_context = 3
    reward_type = "binary"
    reward_structure = "cascade_additive"
    evaluation_policy_logit_ = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [3, 4, 5, 6]])

    dataset_none = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=None,
        random_state=12345,
        base_reward_function=logistic_reward_function,
        is_factorizable=is_factorizable,
    )
    logged_bandit_feedback_none = dataset_none.obtain_batch_bandit_feedback(
        n_rounds=n_rounds
    )
    policy_value_none = dataset_none.calc_ground_truth_policy_value(
        evaluation_policy_logit_=evaluation_policy_logit_,
        context=logged_bandit_feedback_none["context"],
    )

    dataset_pbm = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model="pbm",
        random_state=12345,
        base_reward_function=logistic_reward_function,
        is_factorizable=is_factorizable,
    )
    logged_bandit_feedback_pbm = dataset_pbm.obtain_batch_bandit_feedback(
        n_rounds=n_rounds
    )
    policy_value_pbm = dataset_pbm.calc_ground_truth_policy_value(
        evaluation_policy_logit_=evaluation_policy_logit_,
        context=logged_bandit_feedback_pbm["context"],
    )

    dataset_cascade = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model="cascade",
        random_state=12345,
        base_reward_function=logistic_reward_function,
        is_factorizable=is_factorizable,
    )
    logged_bandit_feedback_cascade = dataset_cascade.obtain_batch_bandit_feedback(
        n_rounds=n_rounds
    )
    policy_value_cascade = dataset_cascade.calc_ground_truth_policy_value(
        evaluation_policy_logit_=evaluation_policy_logit_,
        context=logged_bandit_feedback_cascade["context"],
    )

    assert policy_value_pbm < policy_value_none
    assert policy_value_cascade < policy_value_none


# "len_list, click_model, is_factorizable"
valid_input_of_calc_ground_truth_policy_value = [
    (3, "pbm", False),
    (3, "pbm", True),
    (3, "cascade", False),
    (3, "cascade", True),
    (5, "cascade", True),
]


@pytest.mark.parametrize(
    "len_list, click_model, is_factorizable",
    valid_input_of_calc_ground_truth_policy_value,
)
def test_calc_ground_truth_policy_value_value_check_with_eta(
    len_list, click_model, is_factorizable
):
    n_rounds = 3
    n_unique_action = 4
    dim_context = 3
    reward_type = "binary"
    reward_structure = "cascade_additive"
    evaluation_policy_logit_ = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [3, 4, 5, 6]])

    dataset_05 = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        eta=0.5,
        random_state=12345,
        base_reward_function=logistic_reward_function,
        is_factorizable=is_factorizable,
    )
    logged_bandit_feedback_05 = dataset_05.obtain_batch_bandit_feedback(
        n_rounds=n_rounds
    )
    policy_value_05 = dataset_05.calc_ground_truth_policy_value(
        evaluation_policy_logit_=evaluation_policy_logit_,
        context=logged_bandit_feedback_05["context"],
    )

    dataset_1 = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        eta=1.0,
        random_state=12345,
        base_reward_function=logistic_reward_function,
        is_factorizable=is_factorizable,
    )
    logged_bandit_feedback_1 = dataset_1.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    policy_value_1 = dataset_1.calc_ground_truth_policy_value(
        evaluation_policy_logit_=evaluation_policy_logit_,
        context=logged_bandit_feedback_1["context"],
    )

    dataset_2 = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        reward_structure=reward_structure,
        click_model=click_model,
        eta=2.0,
        random_state=12345,
        base_reward_function=logistic_reward_function,
        is_factorizable=is_factorizable,
    )
    logged_bandit_feedback_2 = dataset_2.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    policy_value_2 = dataset_2.calc_ground_truth_policy_value(
        evaluation_policy_logit_=evaluation_policy_logit_,
        context=logged_bandit_feedback_2["context"],
    )

    assert policy_value_2 < policy_value_1 < policy_value_05


n_rounds = 10
n_unique_action = 5
len_list = 3
# action, evaluation_policy_logit_, err, description
invalid_input_of_obtain_pscore_given_evaluation_policy_logit = [
    (
        np.ones((n_rounds, len_list)),
        np.ones((n_rounds, n_unique_action)),
        ValueError,
        "action must be 1-dimensional",
    ),
    (
        np.ones((n_rounds * len_list)),
        np.ones((n_rounds * n_unique_action)),
        ValueError,
        "evaluation_policy_logit_ must be 2-dimensional",
    ),
    (
        np.ones((n_rounds * len_list + 1)),
        np.ones((n_rounds, n_unique_action)),
        ValueError,
        "the shape of action and evaluation_policy_logit_ must be",
    ),
    (
        np.ones((n_rounds * len_list)),
        np.ones((n_rounds, n_unique_action + 1)),
        ValueError,
        "the shape of action and evaluation_policy_logit_ must be",
    ),
    (
        np.ones((n_rounds * len_list)),
        np.ones((n_rounds + 1, n_unique_action)),
        ValueError,
        "the shape of action and evaluation_policy_logit_ must be",
    ),
]


@pytest.mark.parametrize(
    "action, evaluation_policy_logit_, err, description",
    invalid_input_of_obtain_pscore_given_evaluation_policy_logit,
)
def test_obtain_pscore_given_evaluation_policy_logit(
    action, evaluation_policy_logit_, err, description
):
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
    )
    with pytest.raises(err, match=f"{description}*"):
        dataset.obtain_pscore_given_evaluation_policy_logit(
            action=action,
            evaluation_policy_logit_=evaluation_policy_logit_,
        )


# n_unique_action, return_pscore_item_position, is_factorizable
valid_input_of_obtain_pscore_given_evaluation_policy_logit = [
    (10, True, True),
    (10, True, False),
    (10, False, True),
    (10, False, False),
    (3, False, True),
]


@pytest.mark.parametrize(
    "n_unique_action, return_pscore_item_position, is_factorizable",
    valid_input_of_obtain_pscore_given_evaluation_policy_logit,
)
def test_obtain_pscore_given_evaluation_policy_logit_value_check(
    n_unique_action,
    return_pscore_item_position,
    is_factorizable,
):
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=5,
        behavior_policy_function=linear_behavior_policy_logit,
        is_factorizable=is_factorizable,
        random_state=12345,
    )
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=2,
        return_pscore_item_position=return_pscore_item_position,
    )
    behavior_and_evaluation_policy_logit_ = dataset.behavior_policy_function(
        context=bandit_feedback["context"],
        action_context=bandit_feedback["action_context"],
        random_state=dataset.random_state,
    )
    (
        evaluation_policy_pscore,
        evaluation_policy_pscore_item_position,
        evaluation_policy_pscore_cascade,
    ) = dataset.obtain_pscore_given_evaluation_policy_logit(
        action=bandit_feedback["action"],
        evaluation_policy_logit_=behavior_and_evaluation_policy_logit_,
        return_pscore_item_position=return_pscore_item_position,
    )
    print(bandit_feedback["pscore"])
    print(evaluation_policy_pscore)

    assert np.allclose(bandit_feedback["pscore"], evaluation_policy_pscore)
    assert np.allclose(
        bandit_feedback["pscore_cascade"], evaluation_policy_pscore_cascade
    )
    assert (
        np.allclose(
            bandit_feedback["pscore_item_position"],
            evaluation_policy_pscore_item_position,
        )
        if return_pscore_item_position
        else bandit_feedback["pscore_item_position"]
        == evaluation_policy_pscore_item_position
    )


# n_unique_action, len_list, all_slate_actions, policy_logit_i_, true_pscores, description
valid_input_of_calc_pscore_given_policy_logit = [
    (
        5,
        3,
        np.array([[0, 1, 2], [3, 1, 0]]),
        np.arange(5),
        np.array(
            [
                [
                    np.exp(0) / np.exp([0, 1, 2, 3, 4]).sum(),
                    np.exp(1) / np.exp([1, 2, 3, 4]).sum(),
                    np.exp(2) / np.exp([2, 3, 4]).sum(),
                ],
                [
                    np.exp(3) / np.exp([0, 1, 2, 3, 4]).sum(),
                    np.exp(1) / np.exp([0, 1, 2, 4]).sum(),
                    np.exp(0) / np.exp([0, 2, 4]).sum(),
                ],
            ]
        ).prod(axis=1),
        "calc pscores of several slate actions",
    ),
]


@pytest.mark.parametrize(
    "n_unique_action, len_list, all_slate_actions, policy_logit_i_, true_pscores, description",
    valid_input_of_calc_pscore_given_policy_logit,
)
def test_calc_pscore_given_policy_logit_using_valid_input_data(
    n_unique_action,
    len_list,
    all_slate_actions,
    policy_logit_i_,
    true_pscores,
    description,
) -> None:
    # set parameters
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
        base_reward_function=logistic_reward_function,
    )
    pscores = dataset._calc_pscore_given_policy_logit(
        all_slate_actions, policy_logit_i_
    )
    assert np.allclose(true_pscores, pscores)
    pscores = dataset._calc_pscore_given_policy_softmax(
        all_slate_actions, np.exp(policy_logit_i_)
    )
    assert np.allclose(true_pscores, pscores)


# n_unique_action, len_list, evaluation_policy_logit_, action, true_pscores, true_pscores_cascade, true_pscores_item_position,description
mock_input_of_obtain_pscore_given_evaluation_policy_logit = [
    (
        3,
        2,
        np.array([[0, 1, 2], [2, 1, 0]]),
        np.array([2, 1, 2, 0]),
        np.repeat(
            np.array(
                [
                    [
                        np.exp(2) / np.exp([0, 1, 2]).sum(),
                        np.exp(1) / np.exp([0, 1]).sum(),
                    ],
                    [
                        np.exp(0) / np.exp([0, 1, 2]).sum(),
                        np.exp(2) / np.exp([1, 2]).sum(),
                    ],
                ]
            ).prod(axis=1),
            2,
        ),
        np.array(
            [
                [
                    np.exp(2) / np.exp([0, 1, 2]).sum(),
                    np.exp(1) / np.exp([0, 1]).sum(),
                ],
                [
                    np.exp(0) / np.exp([0, 1, 2]).sum(),
                    np.exp(2) / np.exp([1, 2]).sum(),
                ],
            ]
        )
        .cumprod(axis=1)
        .flatten(),
        np.array(
            [
                [
                    np.exp(2)
                    / np.exp([0, 1, 2]).sum()
                    * np.exp(1)
                    / np.exp([0, 1]).sum(),
                    np.exp(2)
                    / np.exp([0, 1, 2]).sum()
                    * np.exp(0)
                    / np.exp([0, 1]).sum(),
                ],
                [
                    np.exp(2)
                    / np.exp([0, 1, 2]).sum()
                    * np.exp(1)
                    / np.exp([0, 1]).sum(),
                    np.exp(0)
                    / np.exp([0, 1, 2]).sum()
                    * np.exp(1)
                    / np.exp([1, 2]).sum(),
                ],
                [
                    np.exp(0)
                    / np.exp([0, 1, 2]).sum()
                    * np.exp(1)
                    / np.exp([1, 2]).sum(),
                    np.exp(0)
                    / np.exp([0, 1, 2]).sum()
                    * np.exp(2)
                    / np.exp([1, 2]).sum(),
                ],
                [
                    np.exp(1)
                    / np.exp([0, 1, 2]).sum()
                    * np.exp(2)
                    / np.exp([0, 2]).sum(),
                    np.exp(0)
                    / np.exp([0, 1, 2]).sum()
                    * np.exp(2)
                    / np.exp([1, 2]).sum(),
                ],
            ]
        ).sum(axis=1),
        "calc three pscores using mock data",
    ),
]


@pytest.mark.parametrize(
    "n_unique_action, len_list, evaluation_policy_logit_, action, true_pscores, true_pscores_cascade, true_pscores_item_position,description",
    mock_input_of_obtain_pscore_given_evaluation_policy_logit,
)
def test_obtain_pscore_given_evaluation_policy_logit_using_mock_input_data(
    n_unique_action,
    len_list,
    evaluation_policy_logit_,
    action,
    true_pscores,
    true_pscores_cascade,
    true_pscores_item_position,
    description,
) -> None:
    # set parameters
    dim_context = 2
    reward_type = "binary"
    random_state = 12345
    dataset = SyntheticSlateBanditDataset(
        n_unique_action=n_unique_action,
        len_list=len_list,
        dim_context=dim_context,
        reward_type=reward_type,
        random_state=random_state,
        base_reward_function=logistic_reward_function,
    )
    (
        evaluation_policy_pscore,
        evaluation_policy_pscore_item_position,
        evaluation_policy_pscore_cascade,
    ) = dataset.obtain_pscore_given_evaluation_policy_logit(
        action, evaluation_policy_logit_, return_pscore_item_position=True
    )
    assert np.allclose(true_pscores, evaluation_policy_pscore)
    assert np.allclose(true_pscores_cascade, evaluation_policy_pscore_cascade)
    assert np.allclose(
        true_pscores_item_position, evaluation_policy_pscore_item_position
    )

    (
        evaluation_policy_pscore,
        evaluation_policy_pscore_item_position,
        evaluation_policy_pscore_cascade,
    ) = dataset.obtain_pscore_given_evaluation_policy_logit(
        action,
        evaluation_policy_logit_,
        return_pscore_item_position=True,
        clip_logit_value=100.0,
    )
    assert np.allclose(true_pscores, evaluation_policy_pscore)
    assert np.allclose(true_pscores_cascade, evaluation_policy_pscore_cascade)
    assert np.allclose(
        true_pscores_item_position, evaluation_policy_pscore_item_position
    )
