# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Bandit Simulator."""
from copy import deepcopy
from typing import Callable, Dict, List
from typing import Union

import numpy as np
import pandas as pd
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

    reward_round_lookup = None
    if bandit_feedback["round_delays"] is not None:
        reward_round_lookup = create_reward_round_lookup(
            bandit_feedback["round_delays"]
        )

    for round_idx, (position_, context_, factual_reward) in tqdm(
        enumerate(
            zip(
                bandit_feedback["position"],
                bandit_feedback["context"],
                bandit_feedback["factual_reward"],
            )
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

        selected_actions_list.append(selected_actions)
        action_ = selected_actions[position_]
        reward_ = factual_reward[action_]

        if bandit_feedback["round_delays"] is None:
            update_policy(policy_, context_, action_, reward_)
        else:
            available_rounds = reward_round_lookup.get(round_idx, [])
            delayed_update_policy(
                available_rounds, bandit_feedback, selected_actions_list, policy_
            )

            if available_rounds:
                del reward_round_lookup[round_idx]

    if bandit_feedback["round_delays"] is not None:
        for round_idx, available_rounds in reward_round_lookup.items():
            delayed_update_policy(
                available_rounds, bandit_feedback, selected_actions_list, policy_
            )

    action_dist = convert_to_action_dist(
        n_actions=policy.n_actions,
        selected_actions=np.array(selected_actions_list),
    )
    return action_dist


def delayed_update_policy(
    available_rounds: List[int],
    bandits_feedback: BanditFeedback,
    selected_actions_list: List[np.ndarray],
    policy_,
) -> None:
    for available_round_idx in available_rounds:
        position_ = bandits_feedback["position"][available_round_idx]
        available_action = selected_actions_list[available_round_idx][position_]
        available_context = bandits_feedback["context"][available_round_idx]
        available_factual_reward = bandits_feedback["factual_reward"][
            available_round_idx
        ][available_action]
        update_policy(
            policy_, available_context, available_action, available_factual_reward
        )


def create_reward_round_lookup(round_delays: np.ndarray) -> Dict[int, List[int]]:
    """Convert an array of round delays to a dict mapping the available rewards for each round.

    Parameters
    ----------
    round_delays: np.ndarray
        A 1-dimensional numpy array containing the deltas representing how many rounds should be between the taken
        action and reward observation.

    Returns
    --------
    reward_round_lookup: Dict
        A dict with the round at which feedback become available as a key and a list with the index of all actions
        for which the reward becomes available in that round.

    """
    rounds = np.arange(len(round_delays))

    reward_round_pdf = pd.DataFrame(
        {"available_at_round": rounds + round_delays, "exposed_at_round": rounds}
    )

    reward_round_lookup = (
        reward_round_pdf.groupby(["available_at_round"])["exposed_at_round"]
        .apply(list)
        .to_dict()
    )

    return reward_round_lookup


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
