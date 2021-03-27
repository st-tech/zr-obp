# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Bandit Simulator."""
from copy import deepcopy
from typing import Callable

from tqdm import tqdm

import numpy as np

from ..utils import check_bandit_feedback_inputs, convert_to_action_dist
from ..types import BanditFeedback, BanditPolicy
from ..policy.policy_type import PolicyType


def run_bandit_simulation(
    bandit_feedback: BanditFeedback, policy: BanditPolicy
) -> np.ndarray:
    """Run an online bandit algorithm on the given logged bandit feedback data.

    Parameters
    ----------
    bandit_feedback: BanditFeedback
        Logged bandit feedback data used in offline bandit simulation.

    policy: BanditPolicy
        Online bandit policy evaluated in offline bandit simulation (i.e., evaluation policy).

    Returns
    --------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities (can be deterministic).

    """
    for key_ in ["action", "position", "reward", "pscore", "context"]:
        if key_ not in bandit_feedback:
            raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")
    check_bandit_feedback_inputs(
        context=bandit_feedback["context"],
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        position=bandit_feedback["position"],
        pscore=bandit_feedback["pscore"],
    )

    policy_ = policy
    selected_actions_list = list()
    dim_context = bandit_feedback["context"].shape[1]
    if bandit_feedback["position"] is None:
        bandit_feedback["position"] = np.zeros_like(
            bandit_feedback["action"], dtype=int
        )
    for action_, reward_, position_, context_ in tqdm(
        zip(
            bandit_feedback["action"],
            bandit_feedback["reward"],
            bandit_feedback["position"],
            bandit_feedback["context"],
        ),
        total=bandit_feedback["n_rounds"],
    ):

        # select a list of actions
        if policy_.policy_type == PolicyType.CONTEXT_FREE:
            selected_actions = policy_.select_action()
        elif policy_.policy_type == PolicyType.CONTEXTUAL:
            selected_actions = policy_.select_action(context_.reshape(1, dim_context))
        action_match_ = action_ == selected_actions[position_]
        # update parameters of a bandit policy
        # only when selected actions&positions are equal to logged actions&positions
        if action_match_:
            if policy_.policy_type == PolicyType.CONTEXT_FREE:
                policy_.update_params(action=action_, reward=reward_)
            elif policy_.policy_type == PolicyType.CONTEXTUAL:
                policy_.update_params(
                    action=action_,
                    reward=reward_,
                    context=context_.reshape(1, dim_context),
                )
        selected_actions_list.append(selected_actions)

    action_dist = convert_to_action_dist(
        n_actions=bandit_feedback["action"].max() + 1,
        selected_actions=np.array(selected_actions_list),
    )
    return action_dist


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
        Logged bandit feedback data used in offline bandit simulation.
        It must contain "expected_rewards".

    reward_sampler: Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function sampling reward for each given action-context pair,
        i.e., :math:`p(r \\mid x, a)`.

    policy: BanditPolicy
        Online bandit policy evaluated in offline bandit simulation (i.e., evaluation policy).

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
