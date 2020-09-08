# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Offline Bandit Simulator."""
from tqdm import tqdm

import numpy as np

from ..utils import check_bandit_feedback_inputs, convert_to_action_dist
from ..types import BanditFeedback, BanditPolicy


def run_bandit_simulation(
    bandit_feedback: BanditFeedback, policy: BanditPolicy
) -> np.ndarray:
    """Run bandit algorithm on logged bandit feedback data.

    Parameters
    ----------
    bandit_feedback: BanditFeedback
        Logged bandit feedback data to be used in offline bandit simulation.

    policy: BanditPolicy
        Bandit policy to be evaluated in offline bandit simulation (i.e., evaluation policy).

    Returns
    --------
    action_dist: array-like shape (n_rounds, n_actions, len_list)
        Distribution over actions, i.e., probability of items being selected at each position (can be deterministic).

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
        if policy_.policy_type == "contextfree":
            selected_actions = policy_.select_action()
        elif policy_.policy_type == "contextual":
            selected_actions = policy_.select_action(context_.reshape(1, dim_context))
        action_match_ = action_ == selected_actions[position_]
        # update parameters of a bandit policy
        # only when selected actions&positions are equal to logged actions&positions
        if action_match_:
            if policy_.policy_type == "contextfree":
                policy_.update_params(action=action_, reward=reward_)
            elif policy_.policy_type == "contextual":
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
