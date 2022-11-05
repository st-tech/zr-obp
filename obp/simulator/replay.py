import numpy as np
import tqdm as tqdm

from obp.policy.policy_type import PolicyType
from obp.simulator.simulator import BanditPolicy
from obp.types import BanditFeedback
from obp.utils import check_bandit_feedback_inputs, convert_to_action_dist


def run_bandit_replay(
    bandit_feedback: BanditFeedback, policy: BanditPolicy
) -> np.ndarray:
    """Run an online bandit algorithm on given logged bandit feedback data using the replay method.

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

    References
    ------------
    Lihong Li, Wei Chu, John Langford, and Xuanhui Wang.
    "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms.", 2011.
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
