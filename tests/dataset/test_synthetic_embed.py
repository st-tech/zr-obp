import numpy as np
import pytest

from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticBanditDatasetWithActionEmbeds


# n_actions, dim_context, reward_type, reward_std, beta, n_cat_per_dim, latent_param_mat_dim, n_cat_dim, p_e_a_param_std, n_unobserved_cat_dim, n_irrelevant_cat_dim, n_deficient_actions, action_context, random_state, err, description
invalid_input_of_init = [
    (
        "3",  #
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        TypeError,
        "n_actions must be an instance of <class 'int'>, not <class 'str'>.",
    ),
    (
        1,  #
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        ValueError,
        "n_actions == 1, must be >= 2.",
    ),
    (
        3,
        "5",  #
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        TypeError,
        "dim_context must be an instance of <class 'int'>, not <class 'str'>.",
    ),
    (
        3,
        0,  #
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        ValueError,
        "dim_context == 0, must be >= 1.",
    ),
    (
        3,
        5,
        "aaa",  #
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        ValueError,
        "'aaa' is not a valid RewardType",
    ),
    (
        3,
        5,
        "binary",
        "1.0",  #
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        TypeError,
        r"reward_std must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        3,
        5,
        "binary",
        -1.0,  #
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        ValueError,
        "reward_std == -1.0, must be >= 0.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        "0.0",  #
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        TypeError,
        r"beta must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1.0,  #
        1,
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        TypeError,
        r"n_cat_per_dim must be an instance of <class 'int'>, not <class 'float'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        0,  #
        1,
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        ValueError,
        r"n_cat_per_dim == 0, must be >= 1.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1.0,  #
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        TypeError,
        r"latent_param_mat_dim must be an instance of <class 'int'>, not <class 'float'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        0,  #
        1,
        1.0,
        0,
        0,
        0,
        None,
        12345,
        ValueError,
        r"latent_param_mat_dim == 0, must be >= 1.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1.0,  #
        1.0,
        0,
        0,
        0,
        None,
        12345,
        TypeError,
        r"n_cat_dim must be an instance of <class 'int'>, not <class 'float'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        0,  #
        1.0,
        0,
        0,
        0,
        None,
        12345,
        ValueError,
        r"n_cat_dim == 0, must be >= 1.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        "1.0",  #
        0,
        0,
        0,
        None,
        12345,
        TypeError,
        r"p_e_a_param_std must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        -1.0,  #
        0,
        0,
        0,
        None,
        12345,
        ValueError,
        r"p_e_a_param_std == -1.0, must be >= 0.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0.0,  #
        0,
        0,
        None,
        12345,
        TypeError,
        r"n_unobserved_cat_dim must be an instance of <class 'int'>, not <class 'float'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        -1,  #
        0,
        0,
        None,
        12345,
        ValueError,
        r"n_unobserved_cat_dim == -1, must be >= 0.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        2,  #
        0,
        0,
        None,
        12345,
        ValueError,
        r"n_unobserved_cat_dim == 2, must be <= 1.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0.0,  #
        0,
        None,
        12345,
        TypeError,
        r"n_irrelevant_cat_dim must be an instance of <class 'int'>, not <class 'float'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        -1,  #
        0,
        None,
        12345,
        ValueError,
        r"n_irrelevant_cat_dim == -1, must be >= 0.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        2,  #
        0,
        None,
        12345,
        ValueError,
        r"n_irrelevant_cat_dim == 2, must be <= 1.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        "0",  #
        None,
        12345,
        TypeError,
        "n_deficient_actions must be an instance of <class 'int'>, not <class 'str'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        1.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        1.0,  #
        None,
        12345,
        TypeError,
        "n_deficient_actions must be an instance of <class 'int'>, not <class 'float'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        10,  #
        None,
        12345,
        ValueError,
        "n_deficient_actions == 10, must be <= 2.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        np.eye(5),  #
        12345,
        ValueError,
        r"Expected `action_context.shape\[0\] == n_actions`, but found it False.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        np.ones((3, 1, 1)),  #
        12345,
        ValueError,
        "`action_context` must be 2D array",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        "np.ones((3, 1, 1))",  #
        12345,
        ValueError,
        "`action_context` must be 2D array",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        np.eye(3),
        None,  #
        ValueError,
        "`random_state` must be given",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        0.0,
        1,
        1,
        1,
        1.0,
        0,
        0,
        0,
        np.eye(3),
        "",  #
        ValueError,
        "'' cannot be used to seed a numpy.random.RandomState instance",
    ),
]


@pytest.mark.parametrize(
    "n_actions, dim_context, reward_type, reward_std, beta, n_cat_per_dim, latent_param_mat_dim, n_cat_dim, p_e_a_param_std, n_unobserved_cat_dim, n_irrelevant_cat_dim, n_deficient_actions, action_context, random_state, err, description",
    invalid_input_of_init,
)
def test_synthetic_init_using_invalid_inputs(
    n_actions,
    dim_context,
    reward_type,
    reward_std,
    beta,
    n_cat_per_dim,
    latent_param_mat_dim,
    n_cat_dim,
    p_e_a_param_std,
    n_unobserved_cat_dim,
    n_irrelevant_cat_dim,
    n_deficient_actions,
    action_context,
    random_state,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = SyntheticBanditDatasetWithActionEmbeds(
            n_actions=n_actions,
            dim_context=dim_context,
            reward_type=reward_type,
            reward_std=reward_std,
            beta=beta,
            n_deficient_actions=n_deficient_actions,
            n_cat_per_dim=n_cat_per_dim,
            latent_param_mat_dim=latent_param_mat_dim,
            n_cat_dim=n_cat_dim,
            p_e_a_param_std=p_e_a_param_std,
            n_unobserved_cat_dim=n_unobserved_cat_dim,
            n_irrelevant_cat_dim=n_irrelevant_cat_dim,
            action_context=action_context,
            random_state=random_state,
        )


def test_synthetic_init():
    # when reward_function is None, expected_reward is randomly sampled in [0, 1]
    # this check includes the test of `sample_contextfree_expected_reward` function
    dataset = SyntheticBanditDatasetWithActionEmbeds(n_actions=2, beta=0)
    assert len(dataset.expected_reward) == 2
    assert np.all(0 <= dataset.expected_reward) and np.all(dataset.expected_reward <= 1)

    # one-hot action_context when None is given
    ohe = np.eye(2, dtype=int)
    assert np.allclose(dataset.action_context, ohe)


# context, action, description
invalid_input_of_sample_reward = [
    ("3", np.ones(2, dtype=int), "`context` must be 2D array"),
    (None, np.ones(2, dtype=int), "`context` must be 2D array"),
    (np.ones((2, 3)), "3", "`action` must be 1D array"),
    (np.ones((2, 3)), None, "`action` must be 1D array"),
    (
        np.ones((2, 3)),
        np.ones(2, dtype=np.float32),
        "the dtype of action must be a subdtype of int",
    ),
    (np.ones(2), np.ones(2, dtype=int), "`context` must be 2D array"),
    (
        np.ones((2, 3)),
        np.ones((2, 3), dtype=int),
        "`action` must be 1D array",
    ),
    (
        np.ones((2, 3)),
        np.ones(3, dtype=int),
        "Expected `context.shape[0]",
    ),
]

valid_input_of_sample_reward = [
    (
        np.ones((2, 3)),
        np.ones(2, dtype=int),
        "valid shape",
    ),
]


@pytest.mark.parametrize(
    "context, action, description",
    invalid_input_of_sample_reward,
)
def test_synthetic_sample_reward_using_invalid_inputs(context, action, description):
    n_actions = 10
    dataset = SyntheticBanditDatasetWithActionEmbeds(n_actions=n_actions)

    with pytest.raises(ValueError, match=f"{description}*"):
        _ = dataset.sample_reward(context=context, action=action)


@pytest.mark.parametrize(
    "context, action, description",
    valid_input_of_sample_reward,
)
def test_synthetic_sample_reward_using_valid_inputs(context, action, description):
    n_actions = 10
    dataset = SyntheticBanditDatasetWithActionEmbeds(n_actions=n_actions, dim_context=3)

    reward = dataset.sample_reward(context=context, action=action)
    assert isinstance(reward, np.ndarray), "Invalid response of sample_reward"
    assert reward.shape == action.shape, "Invalid response of sample_reward"


def test_synthetic_obtain_batch_bandit_feedback():
    # n_rounds
    with pytest.raises(ValueError):
        dataset = SyntheticBanditDatasetWithActionEmbeds(n_actions=2)
        dataset.obtain_batch_bandit_feedback(n_rounds=0)

    with pytest.raises(TypeError):
        dataset = SyntheticBanditDatasetWithActionEmbeds(n_actions=2)
        dataset.obtain_batch_bandit_feedback(n_rounds="3")

    # bandit feedback
    n_rounds = 10
    n_actions = 5
    n_cat_dim = 3
    n_cat_per_dim = 5
    for n_deficient_actions in [0, 2]:
        dataset = SyntheticBanditDatasetWithActionEmbeds(
            n_actions=n_actions,
            beta=0,
            n_cat_per_dim=n_cat_per_dim,
            n_cat_dim=n_cat_dim,
            reward_function=logistic_reward_function,
            n_deficient_actions=n_deficient_actions,
        )
        bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        assert bandit_feedback["n_rounds"] == n_rounds
        assert bandit_feedback["n_actions"] == n_actions
        assert (
            bandit_feedback["context"].shape[0] == n_rounds  # n_rounds
            and bandit_feedback["context"].shape[1] == 1  # default dim_context
        )
        assert (
            bandit_feedback["action_context"].shape[0] == n_actions
            and bandit_feedback["action_context"].shape[1] == n_cat_dim
        )
        assert (
            bandit_feedback["action_embed"].shape[0] == n_rounds
            and bandit_feedback["action_embed"].shape[1] == n_cat_dim
        )
        assert (
            bandit_feedback["action"].ndim == 1
            and len(bandit_feedback["action"]) == n_rounds
        )
        assert bandit_feedback["position"] is None
        assert (
            bandit_feedback["reward"].ndim == 1
            and len(bandit_feedback["reward"]) == n_rounds
        )
        assert (
            bandit_feedback["expected_reward"].shape[0] == n_rounds
            and bandit_feedback["expected_reward"].shape[1] == n_actions
        )
        assert (
            bandit_feedback["q_x_e"].shape[0] == n_rounds
            and bandit_feedback["q_x_e"].shape[1] == n_cat_per_dim
            and bandit_feedback["q_x_e"].shape[2] == n_cat_dim
        )
        assert (
            bandit_feedback["p_e_a"].shape[0] == n_actions
            and bandit_feedback["p_e_a"].shape[1] == n_cat_per_dim
            and bandit_feedback["p_e_a"].shape[2] == n_cat_dim
        )
        assert (
            bandit_feedback["pi_b"].shape[0] == n_rounds
            and bandit_feedback["pi_b"].shape[1] == n_actions
        )
        # when `beta=0`, behavior_policy should be uniform
        if n_deficient_actions == 0:
            uniform_policy = np.ones_like(bandit_feedback["pi_b"]) / n_actions
            assert np.allclose(bandit_feedback["pi_b"], uniform_policy)
        assert np.allclose(bandit_feedback["pi_b"][:, :, 0].sum(1), np.ones(n_rounds))
        assert (bandit_feedback["pi_b"] == 0).sum() == n_deficient_actions * n_rounds
        assert (
            bandit_feedback["pscore"].ndim == 1
            and len(bandit_feedback["pscore"]) == n_rounds
        )


# expected_reward, action_dist, description
invalid_input_of_calc_policy_value = [
    (
        np.ones((2, 3)),
        np.ones((3, 3, 3)),
        "Expected `expected_reward.shape[0]",
    ),
    (
        np.ones((2, 3)),
        np.ones((2, 2, 3)),
        "Expected `expected_reward.shape[1]",
    ),
    ("3", np.ones((2, 2, 3)), "`expected_reward` must be 2D array"),
    (None, np.ones((2, 2, 3)), "`expected_reward` must be 2D array"),
    (np.ones((2, 3)), np.ones((2, 3)), "`action_dist` must be 3D array"),
    (np.ones((2, 3)), "3", "`action_dist` must be 3D array"),
    (np.ones((2, 3)), None, "`action_dist` must be 3D array"),
]

valid_input_of_calc_policy_value = [
    (
        np.ones((2, 3)),
        np.ones((2, 3, 1)),
        "valid shape",
    ),
]


@pytest.mark.parametrize(
    "expected_reward, action_dist, description",
    invalid_input_of_calc_policy_value,
)
def test_synthetic_calc_policy_value_using_invalid_inputs(
    expected_reward,
    action_dist,
    description,
):
    n_actions = 10
    dataset = SyntheticBanditDatasetWithActionEmbeds(n_actions=n_actions)

    with pytest.raises(ValueError, match=f"{description}*"):
        _ = dataset.calc_ground_truth_policy_value(
            expected_reward=expected_reward, action_dist=action_dist
        )


@pytest.mark.parametrize(
    "expected_reward, action_dist, description",
    valid_input_of_calc_policy_value,
)
def test_synthetic_calc_policy_value_using_valid_inputs(
    expected_reward,
    action_dist,
    description,
):
    n_actions = 10
    dataset = SyntheticBanditDatasetWithActionEmbeds(n_actions=n_actions)

    policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=expected_reward, action_dist=action_dist
    )
    assert isinstance(
        policy_value, float
    ), "Invalid response of calc_ground_truth_policy_value"
