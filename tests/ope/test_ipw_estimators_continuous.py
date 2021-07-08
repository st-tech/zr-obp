import pytest
import numpy as np

from obp.ope import (
    KernelizedInverseProbabilityWeighting,
    KernelizedSelfNormalizedInverseProbabilityWeighting,
)
from obp.dataset import (
    SyntheticContinuousBanditDataset,
    linear_reward_funcion_continuous,
    linear_behavior_policy_continuous,
    linear_synthetic_policy_continuous,
)


def test_synthetic_init():
    # kernel
    with pytest.raises(ValueError):
        KernelizedInverseProbabilityWeighting(kernel="a", bandwidth=0.1)

    with pytest.raises(ValueError):
        KernelizedInverseProbabilityWeighting(kernel=None, bandwidth=0.1)

    with pytest.raises(ValueError):
        KernelizedSelfNormalizedInverseProbabilityWeighting(kernel="a", bandwidth=0.1)

    with pytest.raises(ValueError):
        KernelizedSelfNormalizedInverseProbabilityWeighting(kernel=None, bandwidth=0.1)

    # bandwidth
    with pytest.raises(TypeError):
        KernelizedInverseProbabilityWeighting(kernel="gaussian", bandwidth="a")

    with pytest.raises(TypeError):
        KernelizedInverseProbabilityWeighting(kernel="gaussian", bandwidth=None)

    with pytest.raises(ValueError):
        KernelizedInverseProbabilityWeighting(kernel="gaussian", bandwidth=-1.0)

    with pytest.raises(TypeError):
        KernelizedSelfNormalizedInverseProbabilityWeighting(
            kernel="gaussian", bandwidth="a"
        )

    with pytest.raises(TypeError):
        KernelizedSelfNormalizedInverseProbabilityWeighting(
            kernel="gaussian", bandwidth=None
        )

    with pytest.raises(ValueError):
        KernelizedSelfNormalizedInverseProbabilityWeighting(
            kernel="gaussian", bandwidth=-1.0
        )


# prepare ipw instances
ipw = KernelizedInverseProbabilityWeighting(kernel="cosine", bandwidth=0.1)
snipw = KernelizedSelfNormalizedInverseProbabilityWeighting(
    kernel="epanechnikov", bandwidth=0.1
)

# --- invalid inputs (all kernelized estimators) ---

# action_by_evaluation_policy, action_by_behavior_policy, reward, pscore, description
invalid_input_of_ipw = [
    (
        None,  #
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=5),
        "action_by_evaluation_policy must be 1-dimensional ndarray",
    ),
    (
        np.ones((5, 1)),  #
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=5),
        "action_by_evaluation_policy must be 1-dimensional ndarray",
    ),
    (
        np.ones(5),
        None,  #
        np.ones(5),
        np.random.uniform(size=5),
        "action_by_behavior_policy must be ndarray",
    ),
    (
        np.ones(5),
        np.ones((5, 1)),  #
        np.ones(5),
        np.random.uniform(size=5),
        "action_by_behavior_policy must be 1-dimensional ndarray",
    ),
    (
        np.ones(5),
        np.ones(5),
        None,  #
        np.random.uniform(size=5),
        "reward must be ndarray",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones((5, 1)),  #
        np.random.uniform(size=5),
        "reward must be 1-dimensional ndarray",
    ),
    (
        np.ones(5),
        np.ones(4),  #
        np.ones(3),  #
        np.random.uniform(size=5),
        "action_by_behavior_policy and reward must be the same size",
    ),
    (
        np.ones(4),  #
        np.ones(5),  #
        np.ones(5),
        np.random.uniform(size=5),
        "action_by_behavior_policy and action_by_evaluation_policy must be the same size",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        None,  #
        "pscore must be ndarray",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=(5, 1)),  #
        "pscore must be 1-dimensional ndarray",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=4),  #
        "action_by_behavior_policy, reward, and pscore must be the same size",
    ),
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.arange(5),  #
        "pscore must be positive",
    ),
]


@pytest.mark.parametrize(
    "action_by_evaluation_policy, action_by_behavior_policy, reward, pscore, description",
    invalid_input_of_ipw,
)
def test_ipw_continuous_using_invalid_input_data(
    action_by_evaluation_policy,
    action_by_behavior_policy,
    reward,
    pscore,
    description: str,
) -> None:
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw.estimate_policy_value(
            action_by_evaluation_policy=action_by_evaluation_policy,
            action_by_behavior_policy=action_by_behavior_policy,
            reward=reward,
            pscore=pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw.estimate_interval(
            action_by_evaluation_policy=action_by_evaluation_policy,
            action_by_behavior_policy=action_by_behavior_policy,
            reward=reward,
            pscore=pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = snipw.estimate_policy_value(
            action_by_evaluation_policy=action_by_evaluation_policy,
            action_by_behavior_policy=action_by_behavior_policy,
            reward=reward,
            pscore=pscore,
        )
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = snipw.estimate_interval(
            action_by_evaluation_policy=action_by_evaluation_policy,
            action_by_behavior_policy=action_by_behavior_policy,
            reward=reward,
            pscore=pscore,
        )


# --- valid inputs (all kernelized estimators) ---

# action_by_evaluation_policy, action_by_behavior_policy, reward, pscore
valid_input_of_ipw = [
    (
        np.ones(5),
        np.ones(5),
        np.ones(5),
        np.random.uniform(size=5),
    ),
]


@pytest.mark.parametrize(
    "action_by_evaluation_policy, action_by_behavior_policy, reward, pscore",
    valid_input_of_ipw,
)
def test_ipw_continuous_using_valid_input_data(
    action_by_evaluation_policy,
    action_by_behavior_policy,
    reward,
    pscore,
) -> None:
    _ = ipw.estimate_policy_value(
        action_by_evaluation_policy=action_by_evaluation_policy,
        action_by_behavior_policy=action_by_behavior_policy,
        reward=reward,
        pscore=pscore,
    )
    _ = ipw.estimate_interval(
        action_by_evaluation_policy=action_by_evaluation_policy,
        action_by_behavior_policy=action_by_behavior_policy,
        reward=reward,
        pscore=pscore,
    )
    _ = snipw.estimate_policy_value(
        action_by_evaluation_policy=action_by_evaluation_policy,
        action_by_behavior_policy=action_by_behavior_policy,
        reward=reward,
        pscore=pscore,
    )
    _ = snipw.estimate_interval(
        action_by_evaluation_policy=action_by_evaluation_policy,
        action_by_behavior_policy=action_by_behavior_policy,
        reward=reward,
        pscore=pscore,
    )


# --- confidence intervals ---
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
    "action_by_evaluation_policy, action_by_behavior_policy, reward, pscore",
    valid_input_of_ipw,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description",
    invalid_input_of_estimate_intervals,
)
def test_estimate_intervals_of_all_estimators_using_invalid_input_data(
    action_by_evaluation_policy,
    action_by_behavior_policy,
    reward,
    pscore,
    alpha,
    n_bootstrap_samples,
    random_state,
    description,
) -> None:
    with pytest.raises(ValueError, match=f"{description}*"):
        _ = ipw.estimate_interval(
            action_by_evaluation_policy=action_by_evaluation_policy,
            action_by_behavior_policy=action_by_behavior_policy,
            reward=reward,
            pscore=pscore,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        _ = snipw.estimate_interval(
            action_by_evaluation_policy=action_by_evaluation_policy,
            action_by_behavior_policy=action_by_behavior_policy,
            reward=reward,
            pscore=pscore,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "action_by_evaluation_policy, action_by_behavior_policy, reward, pscore",
    valid_input_of_ipw,
)
@pytest.mark.parametrize(
    "alpha, n_bootstrap_samples, random_state, description",
    valid_input_of_estimate_intervals,
)
def test_estimate_intervals_of_all_estimators_using_valid_input_data(
    action_by_evaluation_policy,
    action_by_behavior_policy,
    reward,
    pscore,
    alpha,
    n_bootstrap_samples,
    random_state,
    description,
) -> None:
    _ = ipw.estimate_interval(
        action_by_evaluation_policy=action_by_evaluation_policy,
        action_by_behavior_policy=action_by_behavior_policy,
        reward=reward,
        pscore=pscore,
        alpha=alpha,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )
    _ = snipw.estimate_interval(
        action_by_evaluation_policy=action_by_evaluation_policy,
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
    # define ipw instances
    ipw = KernelizedInverseProbabilityWeighting(kernel=kernel, bandwidth=0.1)
    snipw = KernelizedSelfNormalizedInverseProbabilityWeighting(
        kernel=kernel, bandwidth=0.1
    )
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
    policy_value_estimated_by_ipw = ipw.estimate_policy_value(
        action_by_evaluation_policy=action_by_evaluation_policy,
        action_by_behavior_policy=action_by_behavior_policy,
        reward=reward,
        pscore=pscore,
    )
    policy_value_estimated_by_snipw = snipw.estimate_policy_value(
        action_by_evaluation_policy=action_by_evaluation_policy,
        action_by_behavior_policy=action_by_behavior_policy,
        reward=reward,
        pscore=pscore,
    )

    # check the performance of OPE
    estimated_policy_value = {
        "ipw": policy_value_estimated_by_ipw,
        "snipw": policy_value_estimated_by_snipw,
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
