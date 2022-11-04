from conftest import generate_action_dist
import numpy as np
import pytest

from obp import ope
from obp.ope import OffPolicyEvaluation
from obp.types import BanditFeedback


# action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, estimated_pscore, estimated_importance_weights, description
invalid_input_of_estimation = [
    (
        None,  #
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        None,
        np.zeros((5, 4, 3)),
        np.ones(5),
        np.ones(5),
        "`action_dist` must be 3D array",
    ),
    (
        generate_action_dist(5, 4, 1)[:, :, 0],  #
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        None,
        np.zeros((5, 4, 1)),
        np.ones(5),
        np.ones(5),
        "`action_dist` must be 3D array",
    ),
    (
        np.ones((5, 4, 3)),  #
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        None,
        np.zeros((5, 4, 3)),
        np.ones(5),
        np.ones(5),
        "`action_dist` must be a probability distribution",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        "4",  #
        np.zeros((5, 4, 3)),
        np.ones(5),
        np.ones(5),
        "`position` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.zeros((5, 4), dtype=int),  #
        np.zeros((5, 4, 3)),
        np.ones(5),
        np.ones(5),
        "`position` must be 1D array",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.zeros(5),  #
        np.zeros((5, 4, 3)),
        np.ones(5),
        np.ones(5),
        "`position` elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.zeros(5, dtype=int) - 1,  #
        np.zeros((5, 4, 3)),
        np.ones(5),
        np.ones(5),
        "`position` elements must be non-negative integers",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.zeros(4, dtype=int),  #
        np.zeros((5, 4, 3)),
        np.ones(5),
        np.ones(5),
        "Expected `position.shape[0]",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.ones(5, dtype=int) * 8,  #
        np.zeros((5, 4, 3)),
        np.ones(5),
        np.ones(5),
        "`position` elements must be smaller than",
    ),
    (
        generate_action_dist(5, 4, 3),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
        np.ones(5),
        None,  #
        np.zeros((5, 4, 3)),
        np.ones(5),
        np.ones(5),
        "`position` elements must be given when",
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
        np.ones(5),
        np.ones(5),
        "all arguments are given and len_list > 1",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.choice(4, size=5),
        np.zeros(5, dtype=int),
        np.ones(5),
        np.zeros(5, dtype=int),
        np.zeros((5, 4, 1)),
        np.ones(5),
        np.ones(5),
        "all arguments are given and len_list == 1",
    ),
    (
        generate_action_dist(5, 4, 1),
        np.random.choice(4, size=5),
        np.zeros(5, dtype=int),
        np.ones(5),
        None,
        np.zeros((5, 4, 1)),
        np.ones(5),
        np.ones(5),
        "position argument is None",
    ),
]


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, estimated_pscore, estimated_importance_weights, description",
    invalid_input_of_estimation,
)
def test_estimation_of_all_estimators_using_invalid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    estimated_rewards_by_reg_model: np.ndarray,
    estimated_pscore: np.ndarray,
    estimated_importance_weights: np.ndarray,
    description: str,
) -> None:
    all_estimators = ope.__all_estimators__
    estimators = [
        getattr(ope.estimators, estimator_name)() for estimator_name in all_estimators
    ]
    all_estimators_tuning = ope.__all_estimators_tuning__
    estimators_tuning = [
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[1, 100, 10000, np.inf],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning
        for tuning_method in ["slope", "mse"]
    ]
    all_estimators_tuning_sg = ope.__all_estimators_tuning_sg__
    estimators_tuning_sg = [
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[0.001, 0.01, 0.1, 1.0],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning_sg
        for tuning_method in ["slope", "mse"]
    ]
    estimators_tuning = estimators_tuning + estimators_tuning_sg
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
                estimated_pscore=estimated_pscore,
                estimated_importance_weights=estimated_importance_weights,
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
                estimated_pscore=estimated_pscore,
                estimated_importance_weights=estimated_importance_weights,
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
                estimated_pscore=estimated_pscore,
            )
            assert est == 0.0, f"policy value must be 0, but {est}"
            assert hasattr(
                estimator_tuning, "best_hyperparam"
            ), "estimator_tuning should have `best_hyperparam` attr"
            if estimator_tuning.tuning_method == "mse":
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
                estimated_pscore=estimated_pscore,
            )
            assert hasattr(
                estimator_tuning, "best_hyperparam"
            ), "estimator_tuning should have `best_hyperparam` attr"
            if estimator_tuning.tuning_method == "mse":
                assert hasattr(
                    estimator_tuning, "estimated_mse_score_dict"
                ), "estimator_tuning should have `estimated_mse_score_dict` attr"


@pytest.mark.parametrize(
    "action_dist, action, reward, pscore, position, estimated_rewards_by_reg_model, estimated_pscore, estimated_importance_weights, description",
    valid_input_of_estimation,
)
def test_estimation_of_all_estimators_using_valid_input_data(
    action_dist: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    pscore: np.ndarray,
    position: np.ndarray,
    estimated_rewards_by_reg_model: np.ndarray,
    estimated_pscore: np.ndarray,
    estimated_importance_weights: np.ndarray,
    description: str,
) -> None:
    all_estimators = ope.__all_estimators__
    estimators = [
        getattr(ope.estimators, estimator_name)() for estimator_name in all_estimators
    ]
    all_estimators_tuning = ope.__all_estimators_tuning__
    estimators_tuning = [
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[1, 100, 10000, np.inf],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning
        for tuning_method in ["slope", "mse"]
    ]
    all_estimators_tuning_sg = ope.__all_estimators_tuning_sg__
    estimators_tuning_sg = [
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[0.001, 0.01, 0.1, 1.0],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning_sg
        for tuning_method in ["slope", "mse"]
    ]
    estimators_tuning = estimators_tuning + estimators_tuning_sg
    # estimate_intervals function raises ValueError of all estimators
    for estimator in estimators:
        _ = estimator.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_importance_weights=estimated_importance_weights,
        )
        _ = estimator.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_importance_weights=estimated_importance_weights,
        )
    for estimator_tuning in estimators_tuning:
        _ = estimator_tuning.estimate_policy_value(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
        )
        _ = estimator_tuning.estimate_interval(
            action_dist=action_dist,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
        )


# alpha, n_bootstrap_samples, random_state, description
invalid_input_of_estimate_intervals = [
    (
        0.05,
        100,
        "s",
        ValueError,
        "'s' cannot be used to seed a numpy.random.RandomState instance",
    ),
    (0.05, -1, 1, ValueError, "n_bootstrap_samples == -1, must be >= 1"),
    (
        0.05,
        "s",
        1,
        TypeError,
        "n_bootstrap_samples must be an instance of <class 'int'>, not <class 'str'>",
    ),
    (-1.0, 1, 1, ValueError, "alpha == -1.0, must be >= 0.0"),
    (2.0, 1, 1, ValueError, "alpha == 2.0, must be <= 1.0"),
    (
        "0",
        1,
        1,
        TypeError,
        "alpha must be an instance of <class 'float'>, not <class 'str'>",
    ),
]

valid_input_of_estimate_intervals = [
    (0.05, 100, 1, "random_state is 1"),
    (0.05, 1, 1, "n_bootstrap_samples is 1"),
]


@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, err, description",
    invalid_input_of_estimate_intervals,
)
def test_estimate_intervals_of_all_estimators_using_invalid_input_data(
    alpha,
    n_bootstrap_samples,
    random_state,
    description: str,
    synthetic_bandit_feedback: BanditFeedback,
    err,
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
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[1, 100, 10000, np.inf],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning
        for tuning_method in ["slope", "mse"]
    ]
    all_estimators_tuning_sg = ope.__all_estimators_tuning_sg__
    estimators_tuning_sg = [
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[0.001, 0.01, 0.1, 1.0],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning_sg
        for tuning_method in ["slope", "mse"]
    ]
    estimators_tuning = estimators_tuning + estimators_tuning_sg
    estimated_pscore = None
    estimated_importance_weights = np.ones(bandit_feedback["action"].shape[0])
    # estimate_intervals function raises ValueError of all estimators
    for estimator in estimators:
        with pytest.raises(err, match=f"{description}*"):
            _ = estimator.estimate_interval(
                reward=bandit_feedback["reward"],
                action=bandit_feedback["action"],
                position=bandit_feedback["position"],
                pscore=bandit_feedback["pscore"],
                action_dist=action_dist,
                estimated_rewards_by_reg_model=expected_reward,
                estimated_pscore=estimated_pscore,
                estimated_importance_weights=estimated_importance_weights,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )
    for estimator_tuning in estimators_tuning:
        with pytest.raises(err, match=f"{description}*"):
            _ = estimator_tuning.estimate_interval(
                reward=bandit_feedback["reward"],
                action=bandit_feedback["action"],
                position=bandit_feedback["position"],
                pscore=bandit_feedback["pscore"],
                action_dist=action_dist,
                estimated_rewards_by_reg_model=expected_reward,
                estimated_pscore=estimated_pscore,
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
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[1, 100, 10000, np.inf],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning
        for tuning_method in ["slope", "mse"]
    ]
    all_estimators_tuning_sg = ope.__all_estimators_tuning_sg__
    estimators_tuning_sg = [
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[0.001, 0.01, 0.1, 1.0],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning_sg
        for tuning_method in ["slope", "mse"]
    ]
    estimators_tuning = estimators_tuning + estimators_tuning_sg
    estimated_pscore = None
    estimated_importance_weights = np.ones(bandit_feedback["action"].shape[0])
    # estimate_intervals function raises ValueError of all estimators
    for estimator in estimators:
        _ = estimator.estimate_interval(
            reward=bandit_feedback["reward"],
            action=bandit_feedback["action"],
            position=bandit_feedback["position"],
            pscore=bandit_feedback["pscore"],
            action_dist=action_dist,
            estimated_rewards_by_reg_model=expected_reward,
            estimated_pscore=estimated_pscore,
            estimated_importance_weights=estimated_importance_weights,
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
            estimated_pscore=estimated_pscore,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


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
    # test most of the estimators (ReplayMethod is not tested because it is out of scope)
    all_estimators = ope.__all_estimators__
    estimators_standard = [
        getattr(ope.estimators, estimator_name)()
        for estimator_name in all_estimators
        if estimator_name not in ["ReplayMethod"]
    ]
    all_estimators_tuning = ope.__all_estimators_tuning__
    estimators_tuning = [
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[1, 100, 10000, np.inf],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning
        for tuning_method in ["slope", "mse"]
    ]
    all_estimators_tuning_sg = ope.__all_estimators_tuning_sg__
    estimators_tuning_sg = [
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[0.001, 0.01, 0.1, 1.0],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning_sg
        for tuning_method in ["slope", "mse"]
    ]
    estimators = estimators_standard + estimators_tuning + estimators_tuning_sg
    # skip estimation
    estimated_pscore = None
    estimated_importance_weights = (
        random_action_dist[
            np.arange(synthetic_bandit_feedback["action"].shape[0]),
            synthetic_bandit_feedback["action"],
            np.zeros(
                synthetic_bandit_feedback["action"].shape[0], dtype=int
            ),  # position is None
        ]
        / synthetic_bandit_feedback["pscore"]
    )
    # conduct OPE
    ope_instance = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=estimators
    )
    estimated_policy_value = ope_instance.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=expected_reward,
        estimated_pscore=estimated_pscore,
        estimated_importance_weights=estimated_importance_weights,
    )
    # check the performance of OPE
    print(f"gt_mean: {gt_mean}")
    for key in estimated_policy_value:
        print(
            f"estimated_value: {estimated_policy_value[key]} ------ estimator: {key}, "
        )
        # test the performance of each estimator
        assert (
            np.abs(gt_mean - estimated_policy_value[key]) / gt_mean <= 0.1
        ), f"OPE of {key} did not work well (relative absolute error is greater than 10%)"


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
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[1, 100, 10000, np.inf],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning
        for tuning_method in ["slope", "mse"]
    ]
    all_estimators_tuning_sg = ope.__all_estimators_tuning_sg__
    estimators_tuning_sg = [
        getattr(ope.estimators_tuning, estimator_name)(
            lambdas=[0.001, 0.01, 0.1, 1.0],
            tuning_method=tuning_method,
        )
        for estimator_name in all_estimators_tuning_sg
        for tuning_method in ["slope", "mse"]
    ]
    estimators = estimators_standard + estimators_tuning + estimators_tuning_sg
    # skip estimation
    estimated_pscore = None
    estimated_importance_weights = (
        random_action_dist[
            np.arange(synthetic_bandit_feedback["action"].shape[0]),
            synthetic_bandit_feedback["action"],
            np.zeros(
                synthetic_bandit_feedback["action"].shape[0], dtype=int
            ),  # position is None
        ]
        / synthetic_bandit_feedback["pscore"]
    )
    # conduct OPE
    ope_instance = OffPolicyEvaluation(
        bandit_feedback=synthetic_bandit_feedback, ope_estimators=estimators
    )
    estimated_policy_value = ope_instance.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=expected_reward,
        estimated_pscore=estimated_pscore,
        estimated_importance_weights=estimated_importance_weights,
    )
    estimated_intervals = ope_instance.estimate_intervals(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=expected_reward,
        estimated_pscore=estimated_pscore,
        estimated_importance_weights=estimated_importance_weights,
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
