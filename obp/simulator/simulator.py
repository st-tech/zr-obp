# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Bandit Simulator."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List, Any
from typing import Union

import numpy as np
from sklearn.utils import check_random_state, check_scalar
from tqdm import tqdm

from ..dataset.reward_type import RewardType
from ..dataset.synthetic import sample_random_uniform_coefficients
from ..policy import BaseContextFreePolicy
from ..policy import BaseContextualPolicy
from ..policy.policy_type import PolicyType
from ..types import BanditFeedback
from ..utils import check_bandit_feedback_inputs, check_array


# bandit policy type
BanditPolicy = Union[BaseContextFreePolicy, BaseContextualPolicy]
coef_func_signature = Callable[
    [np.ndarray, np.ndarray, np.random.RandomState],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


@dataclass
class BanditRound:
    n_actions: int
    context: np.ndarray
    action_context: np.ndarray
    rewards: np.ndarray
    expected_rewards: np.ndarray
    round_delays: np.ndarray


@dataclass
class BanditRounds:
    n_rounds: int
    n_actions: int
    context: np.ndarray
    action_context: np.ndarray
    rewards: np.ndarray
    expected_rewards: np.ndarray
    round_delays: np.ndarray

    def _get_bandit_round(self) -> BanditRound:
        if self.round_delays is not None:
            round_delays = self.round_delays[self.idx]
        else:
            round_delays = None

        return BanditRound(
            n_actions=self.n_actions,
            context=self.context[self.idx],
            action_context=self.action_context,
            rewards=self.rewards[self.idx],
            expected_rewards=self.expected_rewards[self.idx],
            round_delays=round_delays,
        )

    def __iter__(self) -> "BanditRounds":
        self.idx = 0
        return self

    def __next__(self) -> BanditRound:
        if self.idx < len(self):
            result = self._get_bandit_round()
            self.idx += 1
            return result
        else:
            raise StopIteration

    def __len__(self) -> int:
        return self.n_rounds


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
        Function defining the delay rounds for each given action-context pair,
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
        self, expected_reward: np.ndarray
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

    def _sample_context(self, size: tuple) -> np.ndarray:
        context = self.random_.normal(size=size)
        return context

    def next_context(self) -> np.ndarray:
        return self._sample_context(size=(1, self.dim_context))

    def next_context_batch(self, n_rounds: int) -> np.ndarray:
        return self._sample_context(size=(n_rounds, self.dim_context))

    def next_bandit_round(self) -> BanditRound:
        context = self.next_context()
        expected_reward_ = self.calc_expected_reward(context)
        rewards = self.sample_reward_given_expected_reward(expected_reward_)

        round_delays = None
        if self.delay_function:
            round_delays = self.delay_function(
                n_rounds=1,
                n_actions=self.n_actions,
                expected_rewards=expected_reward_,
            )[0]

        return BanditRound(
            n_actions=self.n_actions,
            context=context,
            action_context=self.action_context[0],
            rewards=rewards[0],
            expected_rewards=expected_reward_[0],
            round_delays=round_delays,
        )

    def next_bandit_round_batch(self, n_rounds: int) -> BanditRounds:
        """Obtain a batch of full-information bandit data. Contains rewards for all arms,

        Parameters
        ----------
        n_rounds: int
            Data size of the synthetic logged bandit data.

        Returns
        ---------
        bandit_feedback: BanditRounds
            A batch of bandit rounds that can be used to simulate an online bandit model

        """
        check_scalar(n_rounds, "n_rounds", int, min_val=1)

        contexts = self.next_context_batch(n_rounds=n_rounds)
        expected_reward_ = self.calc_expected_reward(contexts)
        rewards = self.sample_reward_given_expected_reward(expected_reward_)

        round_delays = None
        if self.delay_function:
            round_delays = self.delay_function(
                n_rounds=contexts.shape[0],
                n_actions=self.n_actions,
                expected_rewards=expected_reward_,
            )

        return BanditRounds(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=contexts,
            action_context=self.action_context,
            rewards=rewards,
            expected_rewards=expected_reward_,
            round_delays=round_delays,
        )


@dataclass
class BanditPolicySimulator:
    policy: Any
    environment: BanditEnvironmentSimulator = None

    # Internals
    reward_round_lookup: defaultdict = None

    # To keep track of for after
    _selected_actions: List[int] = None
    _obtained_rewards: List[int] = None
    _ground_truth_rewards: List[np.ndarray] = None
    _contexts: List[np.ndarray] = None
    total_reward: int = 0
    rounds_played: int = 0
    current_round: BanditRound = None

    @property
    def selected_actions(self) -> np.ndarray:
        return np.asarray(self._selected_actions)

    @property
    def obtained_rewards(self) -> np.ndarray:
        return np.asarray(self._obtained_rewards)

    @property
    def ground_truth_rewards(self) -> np.ndarray:
        return np.vstack(self._ground_truth_rewards)

    @property
    def contexts(self) -> np.ndarray:
        return np.vstack(self._contexts)

    def __post_init__(self):
        self._selected_actions = []
        self._obtained_rewards = []
        self._ground_truth_rewards = []
        self._contexts = []
        self.reward_round_lookup = defaultdict(list)

    def start_next_bandit_round(self, bandit_round: BanditRound = None) -> None:
        if not bandit_round:
            self.current_round = self.environment.next_bandit_round()
        else:
            self.current_round = bandit_round

        self.append_contexts(self.current_round.context)
        self.append_ground_truth_rewards(self.current_round.rewards)

    def append_ground_truth_rewards(self, ground_truth_rewards):
        self._ground_truth_rewards.append(ground_truth_rewards)

    def append_contexts(self, context):
        self._contexts.append(context)

    def step(self, bandit_round: BanditRound = None):
        self.start_next_bandit_round(bandit_round)
        self._step()

    def _step(self):
        selected_action = self.select_action()
        self._selected_actions.append(selected_action)

        reward_ = self.current_round.rewards[selected_action]
        self._obtained_rewards.append(reward_)
        self.total_reward += reward_

        delays = self.current_round.round_delays
        if delays is None:
            self.update_policy(self.current_round.context, selected_action, reward_)
        else:
            # Add the current round to the lookup
            round_delay = delays[selected_action]
            self.reward_round_lookup[round_delay + self.rounds_played].append(
                self.rounds_played
            )

            # Update policy with all available rounds
            available_rounds = self.reward_round_lookup.get(self.rounds_played, [])
            self.delayed_update_policy(available_rounds, self.rounds_played)

        self.rounds_played += 1

    def select_action(self):
        if self.policy.policy_type == PolicyType.CONTEXT_FREE:
            selected_action = self.policy.select_action()[0]
        elif self.policy.policy_type == PolicyType.CONTEXTUAL:
            selected_action = self.policy.select_action(
                np.expand_dims(self.current_round.context, axis=0)
            )[0]
        else:
            raise RuntimeError(
                f"Policy type {self.policy.policy_type} of policy {self.policy.policy_name} is unsupported"
            )
        return selected_action

    def steps(
        self, n_rounds: int = None, batch_bandit_rounds: BanditRounds = None
    ) -> None:
        if n_rounds:
            for _ in tqdm(range(n_rounds)):
                self.step()
        if batch_bandit_rounds:
            for bandit_round in tqdm(batch_bandit_rounds):
                self.step(bandit_round)

    def delayed_update_policy(
        self, available_rounds: List[int], current_round: int
    ) -> None:
        for available_round_idx in available_rounds:
            available_action = self._selected_actions[available_round_idx]
            available_context = self._contexts[available_round_idx]
            available_rewards = self._obtained_rewards[available_round_idx]

            self.update_policy(available_context, available_action, available_rewards)

        self.reward_round_lookup.pop(current_round, None)

    def clear_delayed_queue(self):
        for round_idx, available_rounds in self.reward_round_lookup.copy().items():
            self.delayed_update_policy(available_rounds, current_round=round_idx)

    def update_policy(self, context: np.ndarray, action: int, reward: int) -> None:
        """Run an online bandit algorithm on the given logged bandit feedback data.

        Parameters
        ----------
        context: np.ndarray
            Context in which the policy observed the reward

        action: int
            Action taken by the policy as defined by the `policy` argument

        reward: int
            Reward observed by the policy as defined by the `policy` argument
        """
        if self.policy.policy_type == PolicyType.CONTEXT_FREE:
            self.policy.update_params(action=action, reward=reward)
        elif self.policy.policy_type == PolicyType.CONTEXTUAL:
            self.policy.update_params(
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
