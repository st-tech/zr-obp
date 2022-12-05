import numpy as np
import pytest

from obp.dataset import linear_behavior_policy_continuous
from obp.dataset import linear_reward_funcion_continuous
from obp.dataset import linear_synthetic_policy_continuous
from obp.dataset import SyntheticContinuousBanditDataset
from obp.ope import KernelizedDoublyRobust


def test_synthetic_init():
    # kernel
    with pytest.raises(ValueError):
        KernelizedDoublyRobust(kernel="a", bandwidth=0.1)

    with pytest.raises(ValueError):
        KernelizedDoublyRobust(kernel=None, bandwidth=0.1)

    # bandwidth
    with pytest.raises(TypeError):
        KernelizedDoublyRobust(kernel="gaussian", bandwidth="a")

    with pytest.raises(TypeError):
        KernelizedDoublyRobust(kernel="gaussian", bandwidth=None)

    with pytest.raises(ValueError):
        KernelizedDoublyRobust(kernel="gaussian", bandwidth=-1.0)


# prepare dr instances
dr = KernelizedDoublyRobust(kernel="cosine", bandwidth=0.1)

# --- invalid inputs (all kernelized estimators) ---

# action_by_evaluation_policy, estimated_rewards_by_reg_model, action_by_behavior_policy, reward, pscore, description
invalid_input_of_dr = [
    (
        None,  #
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=5),
        "`action_by_evaluation_policy` must be 1D array",
    ),
    (
        np.ones((5, 1)),  #
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=5),
        "`action_by_evaluation_policy` must be 1D array",
    ),
    (
        np.ones(5),
        None,  #
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=5),
        "`estimated_rewards_by_reg_model` must be 1D array",
    ),
    (
        np.ones(5),
        np.ones((5, 1)),  #
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=5),
        "`estimated_rewards_by_reg_model` must be 1D array",
    ),
    (
        np.ones(5),  #
        np.ones(4),  #
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=5),
        "Expected `estimated_rewards_by_reg_model.shape[0]",
    ),
    (
        np.ones(5),
        np.ones(5),
        None,  #
        np.ones(5),
        np.random.uniform(size=5),
        "`action_by_behavior_policy` must be 1D array",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones((5, 1)),  #
        np.ones(5),
        np.random.uniform(size=5),
        "`action_by_behavior_policy` must be 1D array",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        None,  #
        np.random.uniform(size=5),
        "`reward` must be 1D array",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.ones((5, 1)),  #
        np.random.uniform(size=5),
        "`reward` must be 1D array",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(4),  #
        np.ones(3),  #
        np.random.uniform(size=5),
        "Expected `action_by_behavior_policy.shape[0]",
    ),
    (
        np.ones(5),  #
        np.ones(5),
        np.ones(4),  #
        np.ones(4),
        np.random.uniform(size=5),
        "Expected `action_by_behavior_policy.shape[0]",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.ones(5),
        None,  #
        "`pscore` must be 1D array",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=(5, 1)),  #
        "`pscore` must be 1D array",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=4),  #
        "Expected `action_by_behavior_policy.shape[0]",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.arange(5),  #
        "`pscore` must be positive",
    ),
]


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, action_by_behavior_policy, reward, pscore, description",
    invalid_input_of_dr,
)
def test_dr_continuous_using_invalid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    action_by_behavior_policy,
    reward,
    pscore,
    description: str,
) -> None:
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = dr.estimate_policy_value(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_by_behavior_policy=action_by_behavior_policy,
            reward=reward,
            pscore=pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = dr.estimate_interval(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_by_behavior_policy=action_by_behavior_policy,
            reward=reward,
            pscore=pscore,
        )


# --- valid inputs (all kernelized estimators) ---

# action_by_evaluation_policy, estimated_rewards_by_reg_model, action_by_behavior_policy, reward, pscore
valid_input_of_dr = [
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=5),
    ),
]


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, action_by_behavior_policy, reward, pscore",
    valid_input_of_dr,
)
def test_dr_continuous_using_valid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    action_by_behavior_policy,
    reward,
    pscore,
) -> None:
    _ = dr.estimate_policy_value(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        action_by_behavior_policy=action_by_behavior_policy,
        reward=reward,
        pscore=pscore,
    )


# --- confidence intervals ---
# alpha, n_bootstrap_samples, random_state, err, description
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
        "n_bootstrap_samples must be an instance of int, not str",
    ),
    (-1.0, 1, 1, ValueError, "alpha == -1.0, must be >= 0.0"),
    (2.0, 1, 1, ValueError, "alpha == 2.0, must be <= 1.0"),
    (
        "0",
        1,
        1,
        TypeError,
        "alpha must be an instance of float, not str",
    ),
]

valid_input_of_estimate_intervals = [
    (0.05, 100, 1, "random_state is 1"),
    (0.05, 1, 1, "n_bootstrap_samples is 1"),
]


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, action_by_behavior_policy, reward, pscore",
    valid_input_of_dr,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, err, description",
    invalid_input_of_estimate_intervals,
)
def test_estimate_intervals_of_all_estimators_using_invalid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    action_by_behavior_policy,
    reward,
    pscore,
    alpha,
    n_bootstrap_samples,
    random_state,
    err,
    description,
) -> None:
    with pytest.raises(err, match=f"{description}*"):
        _ = dr.estimate_interval(
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            action_by_behavior_policy=action_by_behavior_policy,
            reward=reward,
            pscore=pscore,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "action_by_evaluation_policy, estimated_rewards_by_reg_model, action_by_behavior_policy, reward, pscore",
    valid_input_of_dr,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description",
    valid_input_of_estimate_intervals,
)
def test_estimate_intervals_of_all_estimators_using_valid_input_data(
    action_by_evaluation_policy,
    estimated_rewards_by_reg_model,
    action_by_behavior_policy,
    reward,
    pscore,
    alpha,
    n_bootstrap_samples,
    random_state,
    description,
) -> None:
    _ = dr.estimate_interval(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        action_by_behavior_policy=action_by_behavior_policy,
        reward=reward,
        pscore=pscore,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )


@pytest.mark.parametrize(
    "kernel",
    ["triangular", "gaussian", "epanechnikov", "cosine"],
)
def test_continuous_ope_performance(kernel):
    # define dr instances
    dr = KernelizedDoublyRobust(kernel=kernel, bandwidth=0.1)
    # set parameters
    dim_context = 2
    reward_noise = 0.1
    random_state = 12345
    n_rounds = 10000
    min_action_value = -10
    max_action_value = 10
    behavior_policy_function = linear_behavior_policy_continuous
    reward_function = linear_reward_funcion_continuous
    dataset = SyntheticContinuousBanditDataset(
        dim_context=dim_context,
        reward_noise=reward_noise,
        min_action_value=min_action_value,
        max_action_value=max_action_value,
        reward_function=reward_function,
        behavior_policy_function=behavior_policy_function,
        random_state=random_state,
    )
    # obtain feedback
    bandit_feedback = dataset.obtain_batch_bandit_feedback(
        n_rounds=n_rounds,
    )
    context = bandit_feedback["context"]
    action_by_evaluation_policy = linear_synthetic_policy_continuous(context)
    action_by_behavior_policy = bandit_feedback["action"]
    reward = bandit_feedback["reward"]
    pscore = bandit_feedback["pscore"]

    # compute statistics of ground truth policy value
    q_pi_e = linear_reward_funcion_continuous(
        context=context, action=action_by_evaluation_policy, random_state=random_state
    )
    true_policy_value = q_pi_e.mean()
    print(f"true_policy_value: {true_policy_value}")

    # OPE
    policy_value_estimated_by_dr = dr.estimate_policy_value(
        action_by_evaluation_policy=action_by_evaluation_policy,
        estimated_rewards_by_reg_model=q_pi_e,
        action_by_behavior_policy=action_by_behavior_policy,
        reward=reward,
        pscore=pscore,
    )

    # check the performance of OPE
    estimated_policy_value = {
        "dr": policy_value_estimated_by_dr,
    }
    for key in estimated_policy_value:
        print(
            f"estimated_value: {estimated_policy_value[key]} ------ estimator: {key}, "
        )
        # test the performance of each estimator
        assert (
            np.abs(true_policy_value - estimated_policy_value[key]) / true_policy_value
            <= 0.1
        ), f"{key} does not work well (relative estimation error is greater than 10%)"
