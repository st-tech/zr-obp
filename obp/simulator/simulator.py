# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Bandit Simulator."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Optional, List
from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state, check_scalar
from tqdm import tqdm

from ..dataset.reward_type import RewardType
from ..dataset.synthetic import sample_random_uniform_coefficients
from ..policy import BaseContextFreePolicy
from ..policy import BaseContextualPolicy
from ..policy.policy_type import PolicyType
from ..types import BanditFeedback
from ..utils import check_bandit_feedback_inputs, check_array
from ..utils import convert_to_action_dist


# bandit policy type
BanditPolicy = Union[BaseContextFreePolicy, BaseContextualPolicy]
coef_func_signature = Callable[
    [np.ndarray, np.ndarray, np.random.RandomState],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


@dataclass
class BanditEnvironmentSimulator:
    """Class for simulating an environment that can be used with bandit algorithms.

    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times,
    we can resample logged bandit data from the same data generating distribution.
    This can be used to estimate confidence intervals of the performances of OPE estimators.


    Parameters
    -----------
    n_actions: int
        Number of actions.

    dim_context: int, default=1
        Number of dimensions of context vectors.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary', rewards are sampled from the Bernoulli distribution.
        When 'continuous', rewards are sampled from the truncated Normal distribution with `scale=1`.
        The mean parameter of the reward distribution is determined by the `reward_function` specified by the next argument.

    reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function defining the expected reward for each given action-context pair,
        i.e., :math:`q: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None, context **independent** expected rewards will be
        sampled from the uniform distribution automatically.

    delay_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function defining the delay rounds  for each given action-context pair,
        If None, the `delay_rounds` key will be omitted from the dataset samples.

    coef_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=sample_random_uniform_coefficients
        Function responsible for providing coefficients to the reward function. By default coefficients are sampled
        as random uniform.

    reward_std: float, default=1.0
        Standard deviation of the reward distribution.
        A larger value leads to a noisier reward distribution.
        This argument is valid only when `reward_type="continuous"`.

    action_context: np.ndarray, default=None
         Vector representation of (discrete) actions.
         If None, one-hot representation will be used.

    random_state: int, default=12345
        Controls the random seed in sampling synthetic bandit data.

    dataset_name: str, default='synthetic_bandit_dataset'
        Name of the dataset.

    Examples
    ----------

    .. code-block:: python

        >>> from obp.dataset import (
            SyntheticBanditDataset,
            logistic_reward_function
        )

        # generate synthetic contextual bandit feedback with 10 actions.
        >>> dataset = SyntheticBanditDataset(
                n_actions=5,
                dim_context=3,
                reward_function=logistic_reward_function,
                beta=3,
                random_state=12345
            )
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=100000)
        >>> bandit_feedback
        {
            'n_rounds': 10000,
            'n_actions': 5,
            'context': array([[-0.20470766,  0.47894334, -0.51943872],
                    [-0.5557303 ,  1.96578057,  1.39340583],
                    [ 0.09290788,  0.28174615,  0.76902257],
                    ...,
                    [ 0.42468038,  0.48214752, -0.57647866],
                    [-0.51595888, -1.58196174, -1.39237837],
                    [-0.74213546, -0.93858948,  0.03919589]]),
            'action_context': array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]]),
            'action': array([4, 2, 0, ..., 0, 0, 3]),
            'position': None,
            'rewards': array([1, 0, 1, ..., 1, 1, 1]),
            'expected_reward': array([[0.58447584, 0.42261239, 0.28884131, 0.40610288, 0.59416389],
                    [0.13543381, 0.06309101, 0.3696813 , 0.69883145, 0.19717306],
                    [0.52369136, 0.30840555, 0.45036116, 0.59873096, 0.4294134 ],
                    ...,
                    [0.68953133, 0.55893616, 0.34955984, 0.45572919, 0.67187002],
                    [0.88247154, 0.76355595, 0.25545932, 0.19939877, 0.78578675],
                    [0.67637136, 0.42096732, 0.33178027, 0.36439361, 0.52300522]]),
        }
    """

    n_actions: int
    dim_context: int = 1
    reward_type: str = RewardType.BINARY.value
    reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    delay_function: Optional[Callable[[int, float], np.ndarray]] = None
    coef_function: Optional[
        Callable[[int, float], np.ndarray]
    ] = sample_random_uniform_coefficients
    reward_std: float = 1.0
    action_context: Optional[np.ndarray] = None
    random_state: int = 12345
    dataset_name: str = "synthetic_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.dim_context, "dim_context", int, min_val=1)

        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)

        if self.reward_function is None:
            self.expected_reward = self.sample_contextfree_expected_reward()

        # one-hot encoding characterizing actions.
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)
        else:
            check_array(
                array=self.action_context, name="action_context", expected_dim=2
            )
            if self.action_context.shape[0] != self.n_actions:
                raise ValueError(
                    "Expected `action_context.shape[0] == n_actions`, but found it False."
                )

    @property
    def len_list(self) -> int:
        """Length of recommendation lists, slate size."""
        return 1

    def sample_contextfree_expected_reward(self) -> np.ndarray:
        """Sample expected reward for each action from the uniform distribution."""
        return self.random_.uniform(size=self.n_actions)

    def calc_expected_reward(self, context: np.ndarray) -> np.ndarray:
        """Sample expected rewards given contexts"""
        # sample reward for each round based on the reward function
        if self.reward_function is None:
            expected_reward_ = np.tile(self.expected_reward, (context.shape[0], 1))
        else:
            expected_reward_ = self.reward_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
                coef_function=self.coef_function,
            )

        return expected_reward_

    def sample_reward_given_expected_reward(
        self,
        expected_reward: np.ndarray
    ) -> np.ndarray:
        return self.random_.binomial(n=1, p=expected_reward)

    def sample_reward(self, context: np.ndarray) -> np.ndarray:
        """Sample rewards given contexts and actions, i.e., :math:`r \\sim p(r | x, a)`.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each data (such as user information).

        action: array-like, shape (n_rounds,)
            Actions chosen by the behavior policy for each context.

        Returns
        ---------
        reward: array-like, shape (n_rounds,)
            Sampled rewards given contexts and actions.

        """
        check_array(array=context, name="context", expected_dim=2)
        expected_reward_ = self.calc_expected_reward(context)
        return self.sample_reward_given_expected_reward(expected_reward_)

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedback:
        """Obtain batch logged bandit data.

        Parameters
        ----------
        n_rounds: int
            Data size of the synthetic logged bandit data.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Synthesized logged bandit data.

        """
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        contexts = self.random_.normal(size=(n_rounds, self.dim_context))

        # calc expected reward given context and action
        expected_reward_ = self.calc_expected_reward(contexts)

        # sample rewards based on the context and action
        rewards = self.sample_reward_given_expected_reward(expected_reward_)

        round_delays = None
        if self.delay_function:
            round_delays = self.delay_function(
                n_rounds=contexts.shape[0],
                n_actions=self.n_actions,
                expected_rewards=expected_reward_,
            )

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=contexts,
            action_context=self.action_context,
            position=None,  # position effect is not considered in synthetic data
            rewards=rewards,
            expected_reward=expected_reward_,
            round_delays=round_delays,
        )


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
    for key_ in ["position", "rewards", "context"]:
        if key_ not in bandit_feedback:
            raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")

    policy_ = policy
    selected_actions_list = list()
    if bandit_feedback["position"] is None:
        bandit_feedback["position"] = np.zeros((bandit_feedback["context"].shape[0]), dtype=int)

    reward_round_lookup = defaultdict(list)
    for round_idx, (position_, context_, rewards_) in tqdm(
        enumerate(
            zip(
                bandit_feedback["position"],
                bandit_feedback["context"],
                bandit_feedback["rewards"],
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
        reward_ = rewards_[action_]

        if bandit_feedback["round_delays"] is None:
            update_policy(policy_, context_, action_, reward_)
        else:
            # Add the current round to the lookup
            round_delay = bandit_feedback["round_delays"][round_idx, action_]
            reward_round_lookup[round_delay + round_idx].append(round_idx)

            # Update policy with all available rounds
            available_rounds = reward_round_lookup.get(round_idx, [])
            delayed_update_policy(
                available_rounds, bandit_feedback, selected_actions_list, policy_
            )

            if available_rounds:
                del reward_round_lookup[round_idx]

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
        available_rewards = bandits_feedback["rewards"][
            available_round_idx
        ][available_action]
        update_policy(
            policy_, available_context, available_action, available_rewards
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


def count_ground_truth_policy_rewards(rewards: np.ndarray, sampled_actions: np.ndarray) -> float:
    """Count the policy rewards of given action distribution on the given .

    Parameters
    -----------
    rewards: array-like, shape (n_rounds, n_actions)
        rewards given context (:math:`x`) and action (:math:`a`),
        This is often the `rewards` entry of the BanditEnvironmentSimulator. The rewards are
        sampled from the expected reward at each round.

    sampled_actions: array-like, shape (n_rounds, n_actions, len_list)
        Action choices of the evaluation policy (should be deterministic).

    Returns
    ----------
    policy_reward: float
        The policy's reward of the given action distribution on the given simulated bandit environment.

    """
    check_array(array=rewards, name="rewards", expected_dim=2)
    check_array(array=sampled_actions, name="action_dist", expected_dim=2)
    if not np.isin(sampled_actions, [0, 1]).all():
        raise ValueError("Expected `sampled_actions` to be binary action choices. ")
    if rewards.shape[0] != sampled_actions.shape[0]:
        raise ValueError(
            "Expected `rewards.shape[0] = action_dist.shape[0]`, but found it False"
        )
    if rewards.shape[1] != sampled_actions.shape[1]:
        raise ValueError(
            "Expected `rewards.shape[1] = action_dist.shape[1]`, but found it False"
        )

    return np.sum(rewards * sampled_actions)