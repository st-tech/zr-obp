from typing import Set

import numpy as np
import pytest
import torch

from obp import ope
from obp.ope import OffPolicyEvaluation
from obp.types import BanditFeedback
from conftest import generate_action_dist


# action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, description
invalid_input_of_estimation = [
    (
        None,  #
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        None,
        np.zeros((5, 4, 3)),
        "action_dist must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 1)[:, :, 0],  #
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        None,
        np.zeros((5, 4, 1)),
        "action_dist.ndim must be 3-dimensional",
    ),
    (
        np.ones((5, 4, 3)),  #
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        None,
        np.zeros((5, 4, 3)),
        "action_dist must be a probability distribution",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        "4",  #
        np.zeros((5, 4, 3)),
        "position must be ndarray",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.zeros((5, 4), dtype=int),  #
        np.zeros((5, 4, 3)),
        "position must be 1-dimensional",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.zeros(5),  #
        np.zeros((5, 4, 3)),
        "position elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.zeros(5, dtype=int) - 1,  #
        np.zeros((5, 4, 3)),
        "position elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.zeros(4, dtype=int),  #
        np.zeros((5, 4, 3)),
        "the first dimension of position and the first dimension of action_dist must be the same.",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.ones(5, dtype=int) * 8,  #
        np.zeros((5, 4, 3)),
        "position elements must be smaller than the third dimension of action_dist",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        None,  #
        np.zeros((5, 4, 3)),
        "position elements must be given when the third dimension of action_dist is greater than 1",
    ),
]

valid_input_of_estimation = [
    (
        generate_action_dist(5, 4, 3),
        np.random.choice(4, size=5),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.random.choice(3, size=5),
        np.zeros((5, 4, 3)),
        "all arguments are given and len_list > 1",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.choice(4, size=5),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.zeros(5, dtype=int),
        np.zeros((5, 4, 1)),
        "all arguments are given and len_list == 1",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.choice(4, size=5),
        np.zeros(5, dtype=int),
        np.ones(5),
        None,
        np.zeros((5, 4, 1)),
        "position argument is None",
    ),
]


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, description",
    invalid_input_of_estimation,
)
def test_estimation_of_all_estimators_using_invalid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    estimated_rewards_by_reg_model: np.ndarray,
    description: str,
) -> None:
    all_estimators = ope.__all_estimators__
    estimators = [
        getattr(ope.estimators, estimator_name)() for estimator_name in all_estimators
    ]
    all_estimators_tuning = ope.__all_estimators_tuning__
    estimators_tuning = [
        getattr(ope.estimators_tuning, estimator_name)([1, 100, 10000, np.inf])
        for estimator_name in all_estimators_tuning
    ]
    # estimate_intervals function raises ValueError of all estimators
    for estimator in estimators:
        with pytest.raises(ValueError, match=f"{description}*"):
            est = estimator.estimate_policy_value(
                action_dist=action_dist,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )
            assert est == 0.0, f"policy value must be 0, but {est}"
        with pytest.raises(ValueError, match=f"{description}*"):
            _ = estimator.estimate_interval(
                action_dist=action_dist,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )

    for estimator_tuning in estimators_tuning:
        with pytest.raises(ValueError, match=f"{description}*"):
            est = estimator_tuning.estimate_policy_value(
                action_dist=action_dist,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )
            assert est == 0.0, f"policy value must be 0, but {est}"
            assert hasattr(
                estimator_tuning, "best_hyperparam"
            ), "estimator_tuning should have `best_hyperparam` attr"
            assert hasattr(
                estimator_tuning, "estimated_mse_score_dict"
            ), "estimator_tuning should have `estimated_mse_score_dict` attr"
        with pytest.raises(ValueError, match=f"{description}*"):
            _ = estimator_tuning.estimate_interval(
                action_dist=action_dist,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )
            assert hasattr(
                estimator_tuning, "best_hyperparam"
            ), "estimator_tuning should have `best_hyperparam` attr"
            assert hasattr(
                estimator_tuning, "estimated_mse_score_dict"
            ), "estimator_tuning should have `estimated_mse_score_dict` attr"


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, description",
    valid_input_of_estimation,
)
def test_estimation_of_all_estimators_using_valid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    estimated_rewards_by_reg_model: np.ndarray,
    description: str,
) -> None:
    all_estimators = ope.__all_estimators__
    estimators = [
        getattr(ope.estimators, estimator_name)() for estimator_name in all_estimators
    ]
    all_estimators_tuning = ope.__all_estimators_tuning__
    estimators_tuning = [
        getattr(ope.estimators_tuning, estimator_name)([1, 100, 10000, np.inf])
        for estimator_name in all_estimators_tuning
    ]
    # estimate_intervals function raises ValueError of all estimators
    for estimator in estimators:
        _ = estimator.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        _ = estimator.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
    for estimator_tuning in estimators_tuning:
        _ = estimator_tuning.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )
        _ = estimator_tuning.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )


# action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, description
invalid_input_of_estimation_tensor = [
    (
        None,  #
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        None,
        torch.from_numpy(np.zeros((5, 4, 3))),
        "action_dist must be Tensor",
    ),
    (
        torch.Tensor(generate_action_dist(5, 4, 1)[:, :, 0]),  #
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        None,
        torch.from_numpy(np.zeros((5, 4, 1))),
        "action_dist.ndim must be 3-dimensional",
    ),
    (
        torch.from_numpy(np.ones((5, 4, 3))),  #
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        None,
        torch.from_numpy(np.zeros((5, 4, 3))),
        "action_dist must be a probability distribution",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        "4",  #
        torch.from_numpy(np.zeros((5, 4, 3))),
        "position must be Tensor",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        torch.from_numpy(np.zeros((5, 4), dtype=int)),  #
        torch.from_numpy(np.zeros((5, 4, 3))),
        "position must be 1-dimensional",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        torch.from_numpy(np.zeros(5)),  #
        torch.from_numpy(np.zeros((5, 4, 3))),
        "position elements must be non-negative integers",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        torch.from_numpy(np.zeros(5, dtype=int) - 1),  #
        torch.from_numpy(np.zeros((5, 4, 3))),
        "position elements must be non-negative integers",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        torch.from_numpy(np.zeros(4, dtype=int)),  #
        torch.from_numpy(np.zeros((5, 4, 3))),
        "the first dimension of position and the first dimension of action_dist must be the same.",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        torch.from_numpy(np.ones(5, dtype=int) * 8),  #
        torch.from_numpy(np.zeros((5, 4, 3))),
        "position elements must be smaller than the third dimension of action_dist",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        None,  #
        torch.from_numpy(np.zeros((5, 4, 3))),
        "position elements must be given when the third dimension of action_dist is greater than 1",
    ),
]

valid_input_of_estimation_tensor = [
    (
        torch.from_numpy(generate_action_dist(5, 4, 3)),
        torch.from_numpy(np.random.choice(4, size=5)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        torch.from_numpy(np.random.choice(3, size=5)),
        torch.from_numpy(np.zeros((5, 4, 3))),
        "all arguments are given and len_list > 1",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 1)),
        torch.from_numpy(np.random.choice(4, size=5)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.zeros((5, 4, 1))),
        "all arguments are given and len_list == 1",
    ),
    (
        torch.from_numpy(generate_action_dist(5, 4, 1)),
        torch.from_numpy(np.random.choice(4, size=5)),
        torch.from_numpy(np.zeros(5, dtype=int)),
        torch.from_numpy(np.ones(5)),
        None,
        torch.from_numpy(np.zeros((5, 4, 1))),
        "position argument is None",
    ),
]


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, description",
    invalid_input_of_estimation_tensor,
)
def test_estimation_tensor_of_all_estimators_using_invalid_input_data(
    action_dist: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    pscore: torch.Tensor,
    position: torch.Tensor,
    estimated_rewards_by_reg_model: torch.Tensor,
    description: str,
) -> None:
    all_estimators = ope.__all_estimators__
    estimators = [
        getattr(ope.estimators, estimator_name)() for estimator_name in all_estimators
    ]
    # estimate_intervals function raises ValueError of all estimators
    for estimator in estimators:
        if estimator.estimator_name in ("rm", "switch-dr"):
            with pytest.raises(NotImplementedError):
                _ = estimator.estimate_policy_value_tensor(
                    action_dist=action_dist,
                    action=action,
                    reward=reward,
                    pscore=pscore,
                    position=position,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                )
        else:
            with pytest.raises(ValueError, match=f"{description}*"):
                est = estimator.estimate_policy_value_tensor(
                    action_dist=action_dist,
                    action=action,
                    reward=reward,
                    pscore=pscore,
                    position=position,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                )
                assert est == 0.0, f"policy value must be 0, but {est}"


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, description",
    valid_input_of_estimation_tensor,
)
def test_estimation_tensor_of_all_estimators_using_valid_input_data(
    action_dist: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    pscore: torch.Tensor,
    position: torch.Tensor,
    estimated_rewards_by_reg_model: torch.Tensor,
    description: str,
) -> None:
    all_estimators = ope.__all_estimators__
    estimators = [
        getattr(ope.estimators, estimator_name)() for estimator_name in all_estimators
    ]
    # estimate_intervals function raises ValueError of all estimators
    for estimator in estimators:
        if estimator.estimator_name in ("rm", "switch-dr"):
            with pytest.raises(NotImplementedError):
                _ = estimator.estimate_policy_value_tensor(
                    action_dist=action_dist,
                    action=action,
                    reward=reward,
                    pscore=pscore,
                    position=position,
                    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                )
        else:
            _ = estimator.estimate_policy_value_tensor(
                action_dist=action_dist,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            )


# alpha, n_bootstrap_samples, random_state, description
invalid_input_of_estimate_intervals = [
    (0.05, 100, "s", "random_state must be an integer"),
    (0.05, -1, 1, "n_bootstrap_samples must be a positive integer"),
    (0.05, "s", 1, "n_bootstrap_samples must be a positive integer"),
    (0.0, 1, 1, "alpha must be a positive float (< 1)"),
    (1.0, 1, 1, "alpha must be a positive float (< 1)"),
    ("0", 1, 1, "alpha must be a positive float (< 1)"),
]

valid_input_of_estimate_intervals = [
    (0.05, 100, 1, "random_state is 1"),
    (0.05, 1, 1, "n_bootstrap_samples is 1"),
]


@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description",
    invalid_input_of_estimate_intervals,
)
def test_estimate_intervals_of_all_estimators_using_invalid_input_data(
    alpha,
    n_bootstrap_samples,
    random_state,
    description: str,
    synthetic_bandit_feedback: BanditFeedback,
    random_action_dist: np.ndarray,
) -> None:
    """
    Test the response of estimate_intervals using invalid data
    """
    bandit_feedback = synthetic_bandit_feedback
    action_dist = random_action_dist
    expected_reward = synthetic_bandit_feedback["expected_reward"][:, :, np.newaxis]
    # test all estimators
    all_estimators = ope.__all_estimators__
    estimators = [
        getattr(ope.estimators, estimator_name)() for estimator_name in all_estimators
    ]
    all_estimators_tuning = ope.__all_estimators_tuning__
    estimators_tuning = [
        getattr(ope.estimators_tuning, estimator_name)([1, 100, 10000, np.inf])
        for estimator_name in all_estimators_tuning
    ]
    # estimate_intervals function raises ValueError of all estimators
    for estimator in estimators:
        with pytest.raises(ValueError, match=f"{description}*"):
            _ = estimator.estimate_interval(
                reward=bandit_feedback["reward"],
                action=bandit_feedback["action"],
                position=bandit_feedback["position"],
                pscore=bandit_feedback["pscore"],
                action_dist=action_dist,
                estimated_rewards_by_reg_model=expected_reward,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )
    for estimator_tuning in estimators_tuning:
        with pytest.raises(ValueError, match=f"{description}*"):
            _ = estimator_tuning.estimate_interval(
                reward=bandit_feedback["reward"],
                action=bandit_feedback["action"],
                position=bandit_feedback["position"],
                pscore=bandit_feedback["pscore"],
                action_dist=action_dist,
                estimated_rewards_by_reg_model=expected_reward,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )


@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description",
    valid_input_of_estimate_intervals,
)
def test_estimate_intervals_of_all_estimators_using_valid_input_data(
    alpha,
    n_bootstrap_samples,
    random_state,
    description: str,
    synthetic_bandit_feedback: BanditFeedback,
    random_action_dist: np.ndarray,
) -> None:
    """
    Test the response of estimate_intervals using valid data
    """
    bandit_feedback = synthetic_bandit_feedback
    action_dist = random_action_dist
    expected_reward = synthetic_bandit_feedback["expected_reward"][:, :, np.newaxis]
    # test all estimators
    all_estimators = ope.__all_estimators__
    estimators = [
        getattr(ope.estimators, estimator_name)() for estimator_name in all_estimators
    ]
    all_estimators_tuning = ope.__all_estimators_tuning__
    estimators_tuning = [
        getattr(ope.estimators_tuning, estimator_name)([1, 100, 10000, np.inf])
        for estimator_name in all_estimators_tuning
    ]
    # estimate_intervals function raises ValueError of all estimators
    for estimator in estimators:
        _ = estimator.estimate_interval(
            reward=bandit_feedback["reward"],
            action=bandit_feedback["action"],
            position=bandit_feedback["position"],
            pscore=bandit_feedback["pscore"],
            action_dist=action_dist,
            estimated_rewards_by_reg_model=expected_reward,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
    for estimator_tuning in estimators_tuning:
        _ = estimator_tuning.estimate_interval(
            reward=bandit_feedback["reward"],
            action=bandit_feedback["action"],
            position=bandit_feedback["position"],
            pscore=bandit_feedback["pscore"],
            action_dist=action_dist,
            estimated_rewards_by_reg_model=expected_reward,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


def test_fixture(
    synthetic_bandit_feedback: BanditFeedback,
    expected_reward_0: np.ndarray,
    feedback_key_set: Set[str],
    random_action_dist: np.ndarray,
) -> None:
    """
    Check the validity of the fixture data generated by conftest.py
    """
    np.testing.assert_array_almost_equal(
        expected_reward_0, synthetic_bandit_feedback["expected_reward"][0]
    )
    assert feedback_key_set == set(
        synthetic_bandit_feedback.keys()
    ), f"Key set of bandit feedback should be {feedback_key_set}, but {synthetic_bandit_feedback.keys()}"


def test_performance_of_ope_estimators_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the performance of ope estimators using synthetic bandit data and random evaluation policy
    """
    expected_reward = synthetic_bandit_feedback["expected_reward"][:, :, np.newaxis]
    action_dist = random_action_dist
    # compute ground truth policy value using expected reward
    q_pi_e = np.average(expected_reward[:, :, 0], weights=action_dist[:, :, 0], axis=1)
    # compute statistics of ground truth policy value
    gt_mean = q_pi_e.mean()
    gt_std = q_pi_e.std(ddof=1)
    # test most of the estimators (ReplayMethod is not tested because it is out of scope)
    all_estimators = ope.__all_estimators__
    estimators_standard = [
        getattr(ope.estimators, estimator_name)()
        for estimator_name in all_estimators
        if estimator_name not in ["ReplayMethod"]
    ]
    all_estimators_tuning = ope.__all_estimators_tuning__
    estimators_tuning = [
        getattr(ope.estimators_tuning, estimator_name)([1, 100, 10000, np.inf])
        for estimator_name in all_estimators_tuning
    ]
    estimators = estimators_standard + estimators_tuning
    # conduct OPE
    ope_instance = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=estimators
    )
    estimated_policy_value = ope_instance.estimate_policy_values(
        action_dist=action_dist, estimated_rewards_by_reg_model=expected_reward
    )
    # check the performance of OPE
    ci_bound = gt_std * 3 / np.sqrt(q_pi_e.shape[0])
    print(f"gt_mean: {gt_mean}, 3 * gt_std / sqrt(n): {ci_bound}")
    for key in estimated_policy_value:
        print(
            f"estimated_value: {estimated_policy_value[key]} ------ estimator: {key}, "
        )
        # test the performance of each estimator
        assert (
            np.abs(gt_mean - estimated_policy_value[key]) <= ci_bound
        ), f"OPE of {key} did not work well (absolute error is greater than 3*sigma)"


def test_response_format_of_ope_estimators_using_random_evaluation_policy(
    synthetic_bandit_feedback: BanditFeedback, random_action_dist: np.ndarray
) -> None:
    """
    Test the response format of ope estimators using synthetic bandit data and random evaluation policy
    """
    expected_reward = synthetic_bandit_feedback["expected_reward"][:, :, np.newaxis]
    action_dist = random_action_dist
    # test all estimators
    all_estimators = ope.__all_estimators__
    estimators_standard = [
        getattr(ope.estimators, estimator_name)() for estimator_name in all_estimators
    ]
    all_estimators_tuning = ope.__all_estimators_tuning__
    estimators_tuning = [
        getattr(ope.estimators_tuning, estimator_name)([1, 100, 10000, np.inf])
        for estimator_name in all_estimators_tuning
    ]
    estimators = estimators_standard + estimators_tuning
    # conduct OPE
    ope_instance = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=estimators
    )
    estimated_policy_value = ope_instance.estimate_policy_values(
        action_dist=action_dist, estimated_rewards_by_reg_model=expected_reward
    )
    estimated_intervals = ope_instance.estimate_intervals(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=expected_reward,
        random_state=12345,
    )
    # check the format of OPE
    for key in estimated_policy_value:
        # check the keys of the output dictionary of the estimate_intervals method
        assert set(estimated_intervals[key].keys()) == set(
            ["mean", "95.0% CI (lower)", "95.0% CI (upper)"]
        ), f"Confidence interval of {key} has invalid keys"
        # check the relationship between the means and the confidence bounds estimated by OPE estimators
        assert (
            estimated_intervals[key]["95.0% CI (lower)"] <= estimated_policy_value[key]
        ) and (
            estimated_intervals[key]["95.0% CI (upper)"] >= estimated_policy_value[key]
        ), f"Estimated policy value of {key} is not included in estimated intervals of that estimator"
        assert (
            estimated_intervals[key]["mean"]
            >= estimated_intervals[key]["95.0% CI (lower)"]
        ), f"Invalid confidence interval of {key}: lower bound > mean"
        assert (
            estimated_intervals[key]["mean"]
            <= estimated_intervals[key]["95.0% CI (upper)"]
        ), f"Invalid confidence interval of {key}: upper bound < mean"
