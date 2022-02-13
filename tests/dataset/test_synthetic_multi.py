import numpy as np
import pytest

from obp.dataset import SyntheticMultiLoggersBanditDataset


# n_actions, dim_context, reward_type, reward_std, betas, rhos, n_deficient_actions, action_context, random_state, err, description
invalid_input_of_init = [
    (
        "3",  #
        5,
        "binary",
        1.0,
        [-1, 0, 1],
        [1, 1, 1],
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
        [-1, 0, 1],
        [1, 1, 1],
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
        [-1, 0, 1],
        [1, 1, 1],
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
        [-1, 0, 1],
        [1, 1, 1],
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
        [-1, 0, 1],
        [1, 1, 1],
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
        [-1, 0, 1],
        [1, 1, 1],
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
        [-1, 0, 1],
        [1, 1, 1],
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
        None,  #
        [1, 1, 1],
        0,
        None,
        12345,
        TypeError,
        "`betas` must be a list, but",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        "None",  #
        [1, 1, 1],
        0,
        None,
        12345,
        TypeError,
        "`betas` must be a list, but",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        ["-1", 0, 1],  #
        [1, 1, 1],
        0,
        None,
        12345,
        TypeError,
        r"betas\[0\] must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        [1, 1, 1],
        None,  #
        0,
        None,
        12345,
        TypeError,
        "`rhos` must be a list, but",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        [1, 1, 1],
        "None",  #
        0,
        None,
        12345,
        TypeError,
        "`rhos` must be a list, but",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        [1, 1, 1],
        ["1", 1, 1],  #
        0,
        None,
        12345,
        TypeError,
        r"rhos\[0\] must be an instance of \(<class 'int'>, <class 'float'>\), not <class 'str'>.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        [1, 1, 1],
        [1, 1, -1],  #
        0,
        None,
        12345,
        ValueError,
        r"rhos\[2\] == -1, must be >= 0.0.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        [-1, 0, 1],  #
        [1, 1, 1, 1],  #
        0,
        None,
        12345,
        ValueError,
        r"Expected `len\(self.betas\) == len\(self.rhos\)`, but Found it False.",
    ),
    (
        3,
        5,
        "binary",
        1.0,
        [-1, 0, 1],
        [1, 1, 1],
        "0",
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
        [-1, 0, 1],
        [1, 1, 1],
        1.0,
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
        [-1, 0, 1],
        [1, 1, 1],
        10,
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
        [-1, 0, 1],
        [1, 1, 1],
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
        [-1, 0, 1],
        [1, 1, 1],
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
        [-1, 0, 1],
        [1, 1, 1],
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
        [-1, 0, 1],
        [1, 1, 1],
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
        [-1, 0, 1],
        [1, 1, 1],
        0,
        np.eye(3),
        "",  #
        ValueError,
        "'' cannot be used to seed a numpy.random.RandomState instance",
    ),
]


@pytest.mark.parametrize(
    "n_actions, dim_context, reward_type, reward_std, betas, rhos, n_deficient_actions, action_context, random_state, err, description",
    invalid_input_of_init,
)
def test_synthetic_multi_init_using_invalid_inputs(
    n_actions,
    dim_context,
    reward_type,
    reward_std,
    betas,
    rhos,
    n_deficient_actions,
    action_context,
    random_state,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = SyntheticMultiLoggersBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            reward_type=reward_type,
            reward_std=reward_std,
            betas=betas,
            rhos=rhos,
            n_deficient_actions=n_deficient_actions,
            action_context=action_context,
            random_state=random_state,
        )


def test_synthetic_obtain_batch_bandit_feedback():
    betas = [-1, 0, 1]
    rhos = [1, 1, 1]
    # n_rounds
    with pytest.raises(ValueError):
        dataset = SyntheticMultiLoggersBanditDataset(
            n_actions=2, betas=betas, rhos=rhos
        )
        dataset.obtain_batch_bandit_feedback(n_rounds=0)

    with pytest.raises(TypeError):
        dataset = SyntheticMultiLoggersBanditDataset(
            n_actions=2, betas=betas, rhos=rhos
        )
        dataset.obtain_batch_bandit_feedback(n_rounds="3")

    # bandit feedback
    n_rounds = 10
    n_actions = 5
    for n_deficient_actions in [0, 2]:
        dataset = SyntheticMultiLoggersBanditDataset(
            n_actions=n_actions,
            betas=betas,
            rhos=rhos,
            n_deficient_actions=n_deficient_actions,
        )
        bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        assert bandit_feedback["n_rounds"] == n_rounds
        assert bandit_feedback["n_actions"] == n_actions
        assert bandit_feedback["n_strata"] == len(betas)
        assert (
            bandit_feedback["context"].shape[0] == n_rounds  # n_rounds
            and bandit_feedback["context"].shape[1] == 1  # default dim_context
        )
        assert (
            bandit_feedback["action_context"].shape[0] == n_actions
            and bandit_feedback["action_context"].shape[1] == n_actions
        )
        assert (
            bandit_feedback["action"].ndim == 1
            and len(bandit_feedback["action"]) == n_rounds
        )
        assert (
            bandit_feedback["stratum_idx"].ndim == 1
            and len(bandit_feedback["stratum_idx"]) == n_rounds
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
            bandit_feedback["pi_b"].shape[0] == n_rounds
            and bandit_feedback["pi_b"].shape[1] == n_actions
        )
        assert np.allclose(bandit_feedback["pi_b"][:, :, 0].sum(1), np.ones(n_rounds))
        assert (bandit_feedback["pi_b"] == 0).sum() == n_deficient_actions * n_rounds
        assert np.allclose(
            bandit_feedback["pi_b_avg"][:, :, 0].sum(1), np.ones(n_rounds)
        )
        assert (
            bandit_feedback["pscore"].ndim == 1
            and len(bandit_feedback["pscore"]) == n_rounds
        )
        assert (
            bandit_feedback["pscore_avg"].ndim == 1
            and len(bandit_feedback["pscore_avg"]) == n_rounds
        )
