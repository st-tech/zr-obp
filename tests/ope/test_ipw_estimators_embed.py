import re

from conftest import generate_action_dist
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from obp.ope import MarginalizedInverseProbabilityWeighting as MIPW
from obp.ope import SelfNormalizedMarginalizedInverseProbabilityWeighting as SNMIPW
from obp.types import BanditFeedback


# n_actions, delta, pi_a_x_e_estimator, embedding_selection_method, min_emb_dim, err, description
invalid_input_of_ipw_init = [
    (
        2.0,  #
        0.05,
        RandomForestClassifier,
        None,
        1,
        TypeError,
        r"n_actions must be an instance of <class 'int'>, not <class 'float'>.",
    ),
    (
        0,  #
        0.05,
        RandomForestClassifier,
        None,
        1,
        ValueError,
        r"n_actions == 0, must be >= 2.",
    ),
    (
        2,
        "0.05",  #
        RandomForestClassifier,
        None,
        1,
        TypeError,
        r"delta must be an instance of <class 'float'>, not <class 'str'>.",
    ),
    (
        2,
        -0.05,  #
        RandomForestClassifier,
        None,
        1,
        ValueError,
        r"delta == -0.05, must be >= 0.0.",
    ),
    (
        2,
        1.05,  #
        RandomForestClassifier,
        None,
        1,
        ValueError,
        r"delta == 1.05, must be <= 1.0.",
    ),
    (
        2,
        0.05,
        None,  #
        None,
        1,
        ValueError,
        r"`pi_a_x_e_estimator` must be a classifier.",
    ),
    (
        2,
        0.05,
        Ridge,  #
        None,
        1,
        ValueError,
        r"`pi_a_x_e_estimator` must be a classifier.",
    ),
    (
        2,
        0.05,
        RandomForestClassifier,
        "None",  #
        1,
        ValueError,
        r"If given, `embedding_selection_method` must be either 'exact' or 'greedy'",
    ),
    (
        2,
        0.05,
        RandomForestClassifier,
        None,
        1.0,  #
        TypeError,
        r"min_emb_dim must be an instance of <class 'int'>, not <class 'float'>.",
    ),
    (
        2,
        0.05,
        RandomForestClassifier,
        None,
        0,
        ValueError,
        r"min_emb_dim == 0, must be >= 1.",
    ),
]


@pytest.mark.parametrize(
    "n_actions, delta, pi_a_x_e_estimator, embedding_selection_method, min_emb_dim, err, description",
    invalid_input_of_ipw_init,
)
def test_mipw_init_using_invalid_inputs(
    n_actions,
    delta,
    pi_a_x_e_estimator,
    embedding_selection_method,
    min_emb_dim,
    err,
    description,
):
    with pytest.raises(err, match=f"{description}*"):
        _ = MIPW(
            n_actions=n_actions,
            delta=delta,
            pi_a_x_e_estimator=pi_a_x_e_estimator,
            embedding_selection_method=embedding_selection_method,
            min_emb_dim=min_emb_dim,
        )

    with pytest.raises(err, match=f"{description}*"):
        _ = SNMIPW(
            n_actions=n_actions,
            delta=delta,
            pi_a_x_e_estimator=pi_a_x_e_estimator,
            embedding_selection_method=embedding_selection_method,
            min_emb_dim=min_emb_dim,
        )


# action_dist, context, action, reward, action_embed, pi_b, p_e_a, position, description
invalid_input_of_mipw = [
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5),  #
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        "`context` must be 2D array",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        None,  #
        np.zeros(5, dtype=int),
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        "`action` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros(5, dtype=int),
        None,  #
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        "`reward` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros(5, dtype=float),  #
        np.zeros(5, dtype=int),
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        "`action` elements must be integers in the range of",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros(5, dtype=int) - 1,  #
        np.zeros(5, dtype=int),
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        "`action` elements must be integers in the range of",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        "4",  #
        np.zeros(5, dtype=int),
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        "`action` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros((3, 2), dtype=int),  #
        np.zeros(5, dtype=int),
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        "`action` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros(5, dtype=int) + 8,  #
        np.zeros(5, dtype=int),
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        r"`action` elements must be integers in the range of`",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros(5, dtype=int),
        "4",  #
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        "`reward` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros(5, dtype=int),
        np.zeros((3, 2), dtype=int),  #
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        "`reward` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros(5, dtype=int),
        np.zeros(4, dtype=int),  #
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        None,
        None,
        "Expected `action.shape[0]",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.random.randn(5),  #
        generate_action_dist(5, 4, 1),
        None,
        None,
        "`action_embed` must be 2D array",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.random.randn(5, 4),
        np.ones((5, 5)),  #
        None,
        None,
        "`pi_b` must be 3D array",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.randn(5, 4),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.random.randn(5, 4),
        generate_action_dist(5, 4, 1),
        np.ones((5, 3)),  #
        None,
        "`p_e_a` must be 3D array",
    ),
]


@pytest.mark.parametrize(
    "action_dist, context, action, reward, action_embed, pi_b, p_e_a, position, description",
    invalid_input_of_mipw,
)
def test_mipw_using_invalid_input_data(
    action_dist: np.ndarray,
    context: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    action_embed: np.ndarray,
    pi_b: np.ndarray,
    p_e_a: np.ndarray,
    position: np.ndarray,
    description: str,
) -> None:
    # prepare ipw instances
    mipw = MIPW(n_actions=2)
    mipw_exact = MIPW(n_actions=2, embedding_selection_method="exact")
    mipw_greedy = MIPW(n_actions=2, embedding_selection_method="greedy")
    snmipw = SNMIPW(n_actions=2)
    for est in [mipw, mipw_exact, mipw_greedy, snmipw]:
        with pytest.raises(ValueError, match=f"{description}*"):
            _ = est.estimate_policy_value(
                action_dist=action_dist,
                context=context,
                action=action,
                reward=reward,
                action_embed=action_embed,
                pi_b=pi_b,
                p_e_a=p_e_a,
                position=position,
            )
        with pytest.raises(ValueError, match=f"{description}*"):
            _ = est.estimate_interval(
                action_dist=action_dist,
                context=context,
                action=action,
                reward=reward,
                action_embed=action_embed,
                pi_b=pi_b,
                p_e_a=p_e_a,
                position=position,
            )


def test_ipw_using_random_evaluation_policy(
    synthetic_bandit_feedback_with_embed: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the format of ipw variants using synthetic bandit data and random evaluation policy
    """
    action_dist = random_action_dist
    # prepare input dict
    input_dict = {
        k: v
        for k, v in synthetic_bandit_feedback_with_embed.items()
        if k in ["reward", "action", "pi_b", "action_embed", "context", "position"]
    }
    input_dict["action_dist"] = action_dist
    mipw = MIPW(n_actions=synthetic_bandit_feedback_with_embed["n_actions"])
    mipw_exact = MIPW(
        n_actions=synthetic_bandit_feedback_with_embed["n_actions"],
        embedding_selection_method="exact",
    )
    mipw_greedy = MIPW(
        n_actions=synthetic_bandit_feedback_with_embed["n_actions"],
        embedding_selection_method="greedy",
    )
    snmipw = SNMIPW(n_actions=synthetic_bandit_feedback_with_embed["n_actions"])
    # ipw estimators can be used without estimated_rewards_by_reg_model
    for estimator in [mipw, mipw_exact, mipw_greedy, snmipw]:
        estimated_policy_value = estimator.estimate_policy_value(**input_dict)
        assert isinstance(
            estimated_policy_value, float
        ), f"invalid type response: {estimator}"

    # remove necessary keys
    del input_dict["reward"]
    del input_dict["action"]
    for estimator in [mipw, snmipw]:
        with pytest.raises(
            TypeError,
            match=re.escape(
                "estimate_policy_value() missing 2 required positional arguments: 'reward' and 'action'"
            ),
        ):
            _ = estimator.estimate_policy_value(**input_dict)
