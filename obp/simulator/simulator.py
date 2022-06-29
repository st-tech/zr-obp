# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Bandit Simulator."""
from copy import deepcopy
from typing import Callable
from typing import Union

import numpy as np
from tqdm import tqdm

from ..policy import BaseContextFreePolicy
from ..policy import BaseContextualPolicy
from ..policy.policy_type import PolicyType
from ..types import BanditFeedback
from ..utils import check_bandit_feedback_inputs
from ..utils import convert_to_action_dist


# bandit policy type
BanditPolicy = Union[BaseContextFreePolicy, BaseContextualPolicy]


def run_bandit_simulation(
    bandit_feedback: BanditFeedback, policy: BanditPolicy
) -> np.ndarray:
    """Run an online bandit algorithm on the given logged bandit feedback data.

    Parameters
    ----------
    bandit_feedback: BanditFeedback
        Logged bandit data used in offline bandit simulation.

    policy: BanditPolicy
        Online bandit policy to be evaluated in offline bandit simulation (i.e., evaluation policy).

    Returns
    --------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities (can be deterministic).

    """
    for key_ in ["position", "reward", "factual_reward", "pscore", "context"]:
        if key_ not in bandit_feedback:
            raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")
    check_bandit_feedback_inputs(
        context=bandit_feedback["context"],
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        factual_reward=bandit_feedback["factual_reward"],
        position=bandit_feedback["position"],
        pscore=bandit_feedback["pscore"],
    )

    policy_ = policy
    selected_actions_list = list()
    if bandit_feedback["position"] is None:
        bandit_feedback["position"] = np.zeros_like(
            bandit_feedback["action"], dtype=int
        )
    for position_, context_, factual_reward in tqdm(
        zip(
            bandit_feedback["position"],
            bandit_feedback["context"],
            bandit_feedback["factual_reward"],
        ),
        total=bandit_feedback["n_rounds"],
    ):

        # select a list of actions
        if policy_.policy_type == PolicyType.CONTEXT_FREE:
            selected_actions = policy_.select_action()
        elif policy_.policy_type == PolicyType.CONTEXTUAL:
            selected_actions = policy_.select_action(np.expand_dims(context_, axis=0))
        else:
            raise RuntimeError(
                f"Policy type {policy_.policy_type} of policy {policy_.policy_name} is unsupported"
            )

        action_ = selected_actions[position_]
        reward_ = factual_reward[action_]

        update_policy(policy_, context_, action_, reward_)
        selected_actions_list.append(selected_actions)

    action_dist = convert_to_action_dist(
        n_actions=policy.n_actions,
        selected_actions=np.array(selected_actions_list),
    )
    return action_dist


def update_policy(
    policy: BanditPolicy, context: np.ndarray, action: int, reward: int
) -> None:
    """Run an online bandit algorithm on the given logged bandit feedback data.

    Parameters
    ----------
    policy: BanditPolicy
        Online bandit policy to be updated.

    context: np.ndarray
        Context in which the policy observed the reward

    action: int
        Action taken by the policy as defined by the `policy` argument

    reward: int
        Reward observed by the policy as defined by the `policy` argument
    """
    if policy.policy_type == PolicyType.CONTEXT_FREE:
        policy.update_params(action=action, reward=reward)
    elif policy.policy_type == PolicyType.CONTEXTUAL:
        policy.update_params(
            action=action,
            reward=reward,
            context=np.expand_dims(context, axis=0),
        )


def calc_ground_truth_policy_value(
    bandit_feedback: BanditFeedback,
    reward_sampler: Callable[[np.ndarray, np.ndarray], float],
    policy: BanditPolicy,
    n_sim: int = 100,
) -> float:
    """Calculate the ground-truth policy value of a given online bandit algorithm by Monte-Carlo Simulation.

    Parameters
    ----------
    bandit_feedback: BanditFeedback
        Logged bandit data used in offline bandit simulation. Must contain "expected_rewards" as a key.

    reward_sampler: Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function sampling reward for each given action-context pair, i.e., :math:`p(r|x, a)`.

    policy: BanditPolicy
        Online bandit policy to be evaluated in offline bandit simulation (i.e., evaluation policy).

    n_sim: int, default=100
        Number of simulations in the Monte Carlo simulation to compute the policy value.

    Returns
    --------
    ground_truth_policy_value: float
        policy value of a given evaluation policy.

    """
    for key_ in [
        "action",
        "position",
        "reward",
        "expected_reward",
        "context",
    ]:
        if key_ not in bandit_feedback:
            raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")
    check_bandit_feedback_inputs(
        context=bandit_feedback["context"],
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        expected_reward=bandit_feedback["expected_reward"],
        position=bandit_feedback["position"],
    )

    cumulative_reward = 0.0
    dim_context = bandit_feedback["context"].shape[1]

    for _ in tqdm(np.arange(n_sim), total=n_sim):
        policy_ = deepcopy(policy)
        for position_, context_, expected_reward_ in zip(
            bandit_feedback["position"],
            bandit_feedback["context"],
            bandit_feedback["expected_reward"],
        ):

            # select a list of actions
            if policy_.policy_type == PolicyType.CONTEXT_FREE:
                selected_actions = policy_.select_action()
            elif policy_.policy_type == PolicyType.CONTEXTUAL:
                selected_actions = policy_.select_action(
                    context_.reshape(1, dim_context)
                )
            action = selected_actions[position_]
            # sample reward
            reward = reward_sampler(
                context_.reshape(1, dim_context), np.array([action])
            )
            cumulative_reward += expected_reward_[action]

            # update parameters of a bandit policy
            if policy_.policy_type == PolicyType.CONTEXT_FREE:
                policy_.update_params(action=action, reward=reward)
            elif policy_.policy_type == PolicyType.CONTEXTUAL:
                policy_.update_params(
                    action=action,
                    reward=reward,
                    context=context_.reshape(1, dim_context),
                )

    return cumulative_reward / (n_sim * bandit_feedback["n_rounds"])
