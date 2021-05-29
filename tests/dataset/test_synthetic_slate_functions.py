import numpy as np
import pytest

from obp.dataset.synthetic import logistic_reward_function
from obp.dataset.synthetic_slate import (
    linear_behavior_policy_logit,
    action_interaction_reward_function,
    generate_symmetric_matrix,
)


def test_generate_symmetric_matrix():
    matrix = generate_symmetric_matrix(n_unique_action=3, random_state=1)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)


# context, action_context, tau, err, description
invalid_input_of_linear_behavior_policy_logit = [
    (
        np.array([1.0, 1.0]),
        np.ones([2, 2]),
        None,
        ValueError,
        "context must be 2-dimensional ndarray",
    ),
    (
        [1.0, 1.0],
        np.ones([2, 2]),
        None,
        ValueError,
        "context must be 2-dimensional ndarray",
    ),
    (
        np.ones([2, 2]),
        np.array([1.0, 1.0]),
        None,
        ValueError,
        "action_context must be 2-dimensional ndarray",
    ),
    (
        np.ones([2, 2]),
        [1.0, 1.0],
        None,
        ValueError,
        "action_context must be 2-dimensional ndarray",
    ),
    (np.ones([2, 2]), np.ones([2, 2]), np.array([1]), TypeError, ""),
    (np.ones([2, 2]), np.ones([2, 2]), -1, ValueError, ""),
]


@pytest.mark.parametrize(
    "context, action_context, tau, err, description",
    invalid_input_of_linear_behavior_policy_logit,
)
def test_linear_behavior_policy_logit_using_invalid_input(
    context, action_context, tau, err, description
):
    if description == "":
        with pytest.raises(err):
            linear_behavior_policy_logit(
                context=context, action_context=action_context, tau=tau
            )
    else:
        with pytest.raises(err, match=f"{description}*"):
            linear_behavior_policy_logit(
                context=context, action_context=action_context, tau=tau
            )


# context, action_context, tau, description
valid_input_of_linear_behavior_policy_logit = [
    (np.ones([2, 2]), np.ones([3, 2]), 1, "valid input"),
]


@pytest.mark.parametrize(
    "context, action_context, tau, description",
    valid_input_of_linear_behavior_policy_logit,
)
def test_linear_behavior_policy_logit_using_valid_input(
    context, action_context, tau, description
):
    logit_value = linear_behavior_policy_logit(
        context=context, action_context=action_context, tau=tau
    )
    assert logit_value.shape == (context.shape[0], action_context.shape[0])


# context, action_context, action, base_reward_function, action_interaction_weight_matrix, reward_type, reward_structure, len_list, is_enumerated, random_state, err, description
invalid_input_of_action_interaction_reward_function = [
    (
        np.array([5, 2]),
        np.ones([4, 2]),
        np.tile(np.arange(3), 5),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "standard_additive",
        3,
        False,
        1,
        ValueError,
        "context must be 2-dimensional ndarray",
    ),
    (
        np.ones([5, 2]),
        np.array([4, 2]),
        np.tile(np.arange(3), 5),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "standard_additive",
        3,
        False,
        1,
        ValueError,
        "action_context must be 2-dimensional ndarray",
    ),
    (
        np.ones([5, 2]),
        np.ones([4, 2]),
        np.ones([5, 2]),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "standard_additive",
        3,
        False,
        1,
        ValueError,
        "action must be 1-dimensional ndarray",
    ),
    (
        np.ones([5, 2]),
        np.ones([4, 2]),
        np.ones(14),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "standard_additive",
        3,
        True,
        1,
        ValueError,
        "the size of axis 0 of context times len_list must be multiple of that of action",
    ),
    (
        np.ones([5, 2]),
        np.ones([4, 2]),
        np.ones(30),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "standard_additive",
        3,
        False,
        1,
        ValueError,
        "the size of axis 0 of context times len_list must be same with that of action",
    ),
    (
        np.ones([5, 2]),
        np.ones([4, 2]),
        np.ones(15),
        logistic_reward_function,
        np.identity(3),
        "aaa",
        "standard_additive",
        3,
        False,
        1,
        ValueError,
        "reward_type must be either",
    ),
    (
        np.ones([5, 2]),
        np.ones([4, 2]),
        np.ones(15),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "aaa",
        3,
        False,
        1,
        ValueError,
        "reward_structure must be either",
    ),
    (
        np.ones([5, 2]),
        np.ones([4, 2]),
        np.ones(15),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "standard_additive",
        3,
        False,
        1,
        ValueError,
        "the shape of action_interaction_weight_matrix must be",
    ),
    (
        np.ones([5, 2]),
        np.ones([4, 2]),
        np.ones(15),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "cascade_additive",
        3,
        False,
        1,
        ValueError,
        "the shape of action_interaction_weight_matrix must be",
    ),
    (
        np.ones([5, 2]),
        np.ones([4, 2]),
        np.ones(15),
        logistic_reward_function,
        np.identity(4),
        "binary",
        "standard_decay",
        3,
        False,
        1,
        ValueError,
        "the shape of action_interaction_weight_matrix must be",
    ),
    (
        np.ones([5, 2]),
        np.ones([4, 2]),
        np.ones(15),
        logistic_reward_function,
        np.identity(4),
        "binary",
        "cascade_decay",
        3,
        False,
        1,
        ValueError,
        "the shape of action_interaction_weight_matrix must be",
    ),
]


@pytest.mark.parametrize(
    "context, action_context, action, base_reward_function, action_interaction_weight_matrix, reward_type, reward_structure, len_list, is_enumerated, random_state, err, description",
    invalid_input_of_action_interaction_reward_function,
)
def test_action_interaction_reward_function_using_invalid_input(
    context,
    action_context,
    action,
    base_reward_function,
    action_interaction_weight_matrix,
    reward_type,
    reward_structure,
    len_list,
    is_enumerated,
    random_state,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = action_interaction_reward_function(
            context=context,
            action_context=action_context,
            action=action,
            action_interaction_weight_matrix=action_interaction_weight_matrix,
            base_reward_function=base_reward_function,
            reward_type=reward_type,
            reward_structure=reward_structure,
            len_list=len_list,
            is_enumerated=is_enumerated,
            random_state=random_state,
        )


# context, action_context, action, base_reward_function, action_interaction_weight_matrix, reward_type, reward_structure, len_list, is_enumerated, random_state, description
valid_input_of_action_interaction_reward_function = [
    (
        np.ones([5, 2]),
        np.eye(4),
        np.tile(np.arange(3), 5),
        logistic_reward_function,
        np.identity(4),
        "binary",
        "standard_additive",
        3,
        False,
        1,
        "binary reward standard additive",
    ),
    (
        np.ones([5, 2]),
        np.eye(4),
        np.tile(np.arange(3), 5),
        logistic_reward_function,
        np.identity(4),
        "binary",
        "cascade_additive",
        3,
        False,
        1,
        "binary reward cascade additive",
    ),
    (
        np.ones([5, 2]),
        np.eye(4),
        np.tile(np.arange(3), 10),
        logistic_reward_function,
        np.identity(4),
        "binary",
        "standard_additive",
        3,
        True,
        1,
        "binary reward standard additive enumerated",
    ),
    (
        np.ones([5, 2]),
        np.eye(4),
        np.tile(np.arange(3), 5),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "standard_decay",
        3,
        False,
        1,
        "binary reward standard decay",
    ),
    (
        np.ones([5, 2]),
        np.eye(4),
        np.tile(np.arange(3), 5),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "cascade_decay",
        3,
        False,
        1,
        "binary reward cascade decay",
    ),
    (
        np.ones([5, 2]),
        np.eye(4),
        np.tile(np.arange(3), 5),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "independent",
        3,
        False,
        1,
        "binary reward independent",
    ),
    (
        np.ones([5, 2]),
        np.eye(4),
        np.tile(np.arange(3), 10),
        logistic_reward_function,
        np.identity(3),
        "binary",
        "standard_decay",
        3,
        True,
        1,
        "binary reward standard decay enumerated",
    ),
    (
        np.ones([5, 2]),
        np.eye(4),
        np.tile(np.arange(3), 5),
        logistic_reward_function,
        np.identity(3),
        "continuous",
        "standard_decay",
        3,
        False,
        1,
        "continuous reward",
    ),
]


@pytest.mark.parametrize(
    "context, action_context, action, base_reward_function, action_interaction_weight_matrix, reward_type, reward_structure, len_list, is_enumerated, random_state, description",
    valid_input_of_action_interaction_reward_function,
)
def test_action_interaction_decay_reward_function_using_valid_input(
    context,
    action_context,
    action,
    base_reward_function,
    action_interaction_weight_matrix,
    reward_type,
    reward_structure,
    len_list,
    is_enumerated,
    random_state,
    description,
):
    expected_reward_factual = action_interaction_reward_function(
        context=context,
        action_context=action_context,
        action=action,
        action_interaction_weight_matrix=action_interaction_weight_matrix,
        base_reward_function=base_reward_function,
        reward_type=reward_type,
        reward_structure=reward_structure,
        len_list=len_list,
        is_enumerated=is_enumerated,
        random_state=random_state,
    )
    if not is_enumerated:
        assert expected_reward_factual.shape == (
            context.shape[0],
            len_list,
        )
    else:
        assert expected_reward_factual.shape[0] % context.shape[0] == 0
        assert expected_reward_factual.shape[1] == len_list
    if reward_type == "binary":
        assert np.all(0 <= expected_reward_factual) and np.all(
            expected_reward_factual <= 1
        )
