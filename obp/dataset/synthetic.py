# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic Logged Bandit Data."""
from collections import deque
from dataclasses import dataclass
from typing import Callable, Tuple
from typing import Optional, List

import numpy as np
from scipy.stats import truncnorm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from ..types import BanditFeedback
from ..utils import check_array
from ..utils import sample_action_fast
from ..utils import sigmoid
from ..utils import softmax
from .base import BaseBanditDataset
from .reward_type import RewardType

coef_func_signature = Callable[
    [np.ndarray, np.ndarray, np.random.RandomState],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


def sample_random_uniform_coefficients(
    effective_dim_action_context: int,
    effective_dim_context: int,
    random_: np.random.RandomState,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    context_coef_ = random_.uniform(-1, 1, size=effective_dim_context)
    action_coef_ = random_.uniform(-1, 1, size=effective_dim_action_context)
    context_action_coef_ = random_.uniform(
        -1, 1, size=(effective_dim_context, effective_dim_action_context)
    )
    return context_coef_, action_coef_, context_action_coef_


@dataclass
class CoefficientDrifter:
    """Class for synthesizing bandit data.

    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times,
    we can resample logged bandit data from the same data generating distribution.
    This can be used to estimate confidence intervals of the performances of OPE estimators.

    If None is given as `behavior_policy_function`, the behavior policy will be generated from the true expected reward function. See the description of the `beta` argument, which controls the behavior policy.

    Parameters
    -----------

    References
    ------------
    Emanuele Cavenaghi, Gabriele Sottocornola, Fabio Stella, and Markus Zanker.
    "Non stationary multi-armed bandit: Empirical evaluation of a new concept drift-aware algorithm.", 2021.

    """

    drift_interval: int
    transition_period: int = 0
    transition_type: str = "linear"  # linear or weighted_sampled
    seasonal: bool = False
    base_coefficient_weight: float = 0.0
    effective_dim_action_context: Optional[int] = None
    effective_dim_context: Optional[int] = None
    random_state: int = 12345

    played_rounds: int = 0
    context_coefs: Optional[deque] = None
    action_coefs: Optional[deque] = None
    context_action_coefs: Optional[deque] = None
    base_context_coef: Optional[np.ndarray] = None
    base_action_coef: Optional[np.ndarray] = None
    base_context_action_coef: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)
        self.available_rounds = self.drift_interval
        self.context_coefs = deque(maxlen=2)
        self.action_coefs = deque(maxlen=2)
        self.context_action_coefs = deque(maxlen=2)
        if self.effective_dim_action_context and self.effective_dim_context:
            self.update_coef()

    def update_coef(self) -> None:
        if self.base_context_coef is None:
            self.base_context_coef, self.base_action_coef, self.base_context_action_coef = sample_random_uniform_coefficients(
                self.effective_dim_action_context,
                self.effective_dim_context,
                self.random_,
            )

        if len(self.context_coefs) == 0:
            self.context_coefs.append(self.base_context_coef)
            self.action_coefs.append(self.base_action_coef)
            self.context_action_coefs.append(self.base_context_action_coef)

        if self.seasonal and len(self.context_coefs) == 2:
            self.context_coefs.rotate()
            self.action_coefs.rotate()
            self.context_action_coefs.rotate()
        else:
            (
                tmp_context_coef,
                tmp_action_coef,
                tmp_action_context_coef,
            ) = sample_random_uniform_coefficients(
                self.effective_dim_action_context,
                self.effective_dim_context,
                self.random_,
            )
            self.context_coefs.append(tmp_context_coef)
            self.action_coefs.append(tmp_action_coef)
            self.context_action_coefs.append(tmp_action_context_coef)

    def get_coefficients(
        self,
        n_rounds: int,
        effective_dim_context: int = None,
        effective_dim_action_context: int = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if effective_dim_action_context and effective_dim_context:
            eff_dim_not_set = (
                not self.effective_dim_action_context and not self.effective_dim_context
            )
            eff_dim_equal = (
                self.effective_dim_action_context == effective_dim_action_context
                and self.effective_dim_context == effective_dim_context
            )
            if eff_dim_not_set or eff_dim_equal:
                self.effective_dim_action_context = effective_dim_action_context
                self.effective_dim_context = effective_dim_context
            else:
                raise RuntimeError("Trying to change the effective dimensions")

        if len(self.context_coefs) == 0:
            self.update_coef()

        required_rounds = n_rounds
        context_coefs = []
        action_coefs = []
        context_action_coefs = []

        while required_rounds > 0:
            if required_rounds >= self.available_rounds:
                self.append_current_coefs(context_coefs, action_coefs, context_action_coefs, rounds=self.available_rounds)
                required_rounds -= self.available_rounds
                self.update_coef()
                self.available_rounds = self.drift_interval
            else:
                self.append_current_coefs(context_coefs, action_coefs, context_action_coefs, rounds=required_rounds)
                self.available_rounds -= required_rounds
                required_rounds = 0

        return (
            np.vstack(context_coefs),
            np.vstack(action_coefs),
            np.vstack(context_action_coefs),
        )

    def append_current_coefs(
        self, context_coefs: List[np.ndarray], action_coefs: List[np.ndarray], context_action_coefs: List[np.ndarray], rounds: int
    ) -> None:
        shift_start = self.available_rounds - self.transition_period

        transition_steps = np.arange(start=1, stop=self.transition_period + 1)
        if shift_start >= 0:
            transition_steps = np.pad(transition_steps, pad_width=[(shift_start, 0)])
        if shift_start < 0:
            transition_steps = transition_steps[-shift_start:]

        shift_remainder = self.available_rounds - rounds
        if shift_remainder > 0:
            transition_steps = transition_steps[shift_remainder:]

        weights = transition_steps / (self.transition_period + 1)

        if self.transition_type is "weighted_sampled":
            weights = self.random_.binomial(n=1, p=weights)

        context_coefs.append(self.compute_weighted_coefs(self.context_coefs, self.base_context_coef, rounds, weights))
        action_coefs.append(self.compute_weighted_coefs(self.action_coefs, self.base_action_coef, rounds, weights))
        context_action_coefs.append(self.compute_weighted_coefs(self.context_action_coefs, self.base_context_action_coef, rounds, weights))


    def compute_weighted_coefs(self, coefs, base_coef, rounds, weights):
        base_coef = self.base_coefficient_weight * base_coef

        A = np.tile(coefs[0], [rounds] + [1 for _ in coefs[0].shape])
        B = np.tile(coefs[1], [rounds] + [1 for _ in coefs[1].shape])
        coefs = (
                base_coef
                + A * np.expand_dims((1 - self.base_coefficient_weight) * (1 - weights), list(range(1, len(A.shape))))
                + B * np.expand_dims((1 - self.base_coefficient_weight) * weights, list(range(1, len(B.shape))))
        )
        return coefs


@dataclass
class SyntheticBanditDataset(BaseBanditDataset):
    """Class for synthesizing bandit data.

    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times,
    we can resample logged bandit data from the same data generating distribution.
    This can be used to estimate confidence intervals of the performances of OPE estimators.

    If None is given as `behavior_policy_function`, the behavior policy will be generated from the true expected reward function. See the description of the `beta` argument, which controls the behavior policy.

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

    behavior_policy_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default=None
        Function generating logit values, which will be used to define the behavior policy via softmax transformation.
        If None, behavior policy will be generated by applying the softmax function to the expected reward.
        Thus, in this case, it is possible to control the optimality of the behavior policy by customizing `beta`.
        If `beta` is large, the behavior policy becomes near-deterministic/near-optimal,
        while a small or negative value of `beta` leads to a sub-optimal behavior policy.

    beta: int or float, default=1.0
        Inverse temperature parameter, which controls the optimality and entropy of the behavior policy.
        A large value leads to a near-deterministic behavior policy,
        while a small value leads to a near-uniform behavior policy.
        A positive value leads to a near-optimal behavior policy,
        while a negative value leads to a sub-optimal behavior policy.

    n_deficient_actions: int, default=0
        Number of deficient actions having zero probability of being selected in the logged bandit data.
        If there are some deficient actions, the full/common support assumption is very likely to be violated,
        leading to some bias for IPW-type estimators. See Sachdeva et al.(2020) for details.
        `n_deficient_actions` should be an integer smaller than `n_actions - 1` so that there exists at least one action
        that have a positive probability of being selected by the behavior policy.

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
            'reward': array([1, 0, 1, ..., 1, 1, 1]),
            'expected_reward': array([[0.58447584, 0.42261239, 0.28884131, 0.40610288, 0.59416389],
                    [0.13543381, 0.06309101, 0.3696813 , 0.69883145, 0.19717306],
                    [0.52369136, 0.30840555, 0.45036116, 0.59873096, 0.4294134 ],
                    ...,
                    [0.68953133, 0.55893616, 0.34955984, 0.45572919, 0.67187002],
                    [0.88247154, 0.76355595, 0.25545932, 0.19939877, 0.78578675],
                    [0.67637136, 0.42096732, 0.33178027, 0.36439361, 0.52300522]]),
            'pi_b': array([[[0.27454777],
                    [0.16342857],
                    [0.12506266],
                    [0.13791739],
                    [0.22195834]]]),
            'pscore': array([0.28264435, 0.19326617, 0.23079467, ..., 0.28729378, 0.36637549,
                    0.13791739])
        }

    References
    ------------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Noveen Sachdeva, Yi Su, and Thorsten Joachims.
    "Off-policy Bandits with Deficient Support.", 2020.

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
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    beta: float = 1.0
    n_deficient_actions: int = 0
    random_state: int = 12345
    dataset_name: str = "synthetic_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.dim_context, "dim_context", int, min_val=1)
        check_scalar(self.beta, "beta", (int, float))
        check_scalar(
            self.n_deficient_actions,
            "n_deficient_actions",
            int,
            min_val=0,
            max_val=self.n_actions - 1,
        )

        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)

        if RewardType(self.reward_type) not in [
            RewardType.BINARY,
            RewardType.CONTINUOUS,
        ]:
            raise ValueError(
                f"`reward_type` must be either '{RewardType.BINARY.value}' or '{RewardType.CONTINUOUS.value}',"
                f"but {self.reward_type} is given.'"
            )
        check_scalar(self.reward_std, "reward_std", (int, float), min_val=0)
        if self.reward_function is None:
            self.expected_reward = self.sample_contextfree_expected_reward()
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            self.reward_min = 0
            self.reward_max = 1e10

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
        expected_reward: np.ndarray,
        action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if action is not None:
            expected_reward = expected_reward[np.arange(action.shape[0]), action]

        """Sample reward given expected rewards"""
        if RewardType(self.reward_type) == RewardType.BINARY:
            reward = self.random_.binomial(n=1, p=expected_reward)
        elif RewardType(self.reward_type) == RewardType.CONTINUOUS:
            mean = expected_reward
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            reward = truncnorm.rvs(
                a=a,
                b=b,
                loc=mean,
                scale=self.reward_std,
                random_state=self.random_state,
            )
        else:
            raise NotImplementedError

        return reward

    def sample_reward(self, context: np.ndarray, action: np.ndarray) -> np.ndarray:
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
        check_array(array=action, name="action", expected_dim=1)
        if context.shape[0] != action.shape[0]:
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0]`, but found it False"
            )
        if not np.issubdtype(action.dtype, np.integer):
            raise ValueError("the dtype of action must be a subdtype of int")

        expected_reward_ = self.calc_expected_reward(context)

        return self.sample_reward_given_expected_reward(expected_reward_, action)

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
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            # correct expected_reward_, as we use truncated normal distribution here
            mean = expected_reward_
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            expected_reward_ = truncnorm.stats(
                a=a, b=b, loc=mean, scale=self.reward_std, moments="m"
            )

        # calculate the action choice probabilities of the behavior policy
        if self.behavior_policy_function is None:
            pi_b_logits = expected_reward_
        else:
            pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        # create some deficient actions based on the value of `n_deficient_actions`
        if self.n_deficient_actions > 0:
            pi_b = np.zeros_like(pi_b_logits)
            n_supported_actions = self.n_actions - self.n_deficient_actions
            supported_actions = np.argsort(
                self.random_.gumbel(size=(n_rounds, self.n_actions)), axis=1
            )[:, ::-1][:, :n_supported_actions]
            supported_actions_idx = (
                np.tile(np.arange(n_rounds), (n_supported_actions, 1)).T,
                supported_actions,
            )
            pi_b[supported_actions_idx] = softmax(
                self.beta * pi_b_logits[supported_actions_idx]
            )
        else:
            pi_b = softmax(self.beta * pi_b_logits)
        # sample actions for each round based on the behavior policy
        actions = sample_action_fast(pi_b, random_state=self.random_state)

        # sample rewards based on the context and action
        factual_reward = self.sample_reward_given_expected_reward(expected_reward_)
        rewards = factual_reward[np.arange(actions.shape[0]), actions]

        round_delays = None
        if self.delay_function:
            round_delays = self.delay_function(
                n_rounds=actions.shape[0],
                n_actions=self.n_actions,
                expected_rewards=expected_reward_,
            )

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=contexts,
            action_context=self.action_context,
            action=actions,
            position=None,  # position effect is not considered in synthetic data
            reward=rewards,
            factual_reward=factual_reward,
            expected_reward=expected_reward_,
            round_delays=round_delays,
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(n_rounds), actions],
        )

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, action_dist: np.ndarray
    ) -> float:
        """Calculate the policy value of given action distribution on the given expected_reward.

        Parameters
        -----------
        expected_reward: array-like, shape (n_rounds, n_actions)
            Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.
            This is often the `expected_reward` of the test set of logged bandit data.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        Returns
        ----------
        policy_value: float
            The policy value of the given action distribution on the given logged bandit data.

        """
        check_array(array=expected_reward, name="expected_reward", expected_dim=2)
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if expected_reward.shape[0] != action_dist.shape[0]:
            raise ValueError(
                "Expected `expected_reward.shape[0] = action_dist.shape[0]`, but found it False"
            )
        if expected_reward.shape[1] != action_dist.shape[1]:
            raise ValueError(
                "Expected `expected_reward.shape[1] = action_dist.shape[1]`, but found it False"
            )

        return np.average(expected_reward, weights=action_dist[:, :, 0], axis=1).mean()

    def count_ground_truth_policy_rewards(
        self, factual_reward: np.ndarray, sampled_actions: np.ndarray
    ) -> float:
        """Count the policy rewards of given action distribution on the given factual_reward.

        Parameters
        -----------
        factual_reward: array-like, shape (n_rounds, n_actions)
            Factual reward given context (:math:`x`) and action (:math:`a`),
            This is often the `factual_reward` of the test set of logged bandit data. The rewards are
            sampled from the expected reward in the logged bandit data.

        sampled_actions: array-like, shape (n_rounds, n_actions, len_list)
            Action choices of the evaluation policy (should be deterministic).

        Returns
        ----------
        policy_reward: float
            The policy's reward of the given action distribution on the given logged bandit data.

        """
        check_array(array=factual_reward, name="expected_reward", expected_dim=2)
        check_array(array=sampled_actions, name="action_dist", expected_dim=2)
        if not np.isin(sampled_actions, [0, 1]).all():
            raise ValueError("Expected `sampled_actions` to be binary action choices. ")
        if factual_reward.shape[0] != sampled_actions.shape[0]:
            raise ValueError(
                "Expected `factual_reward.shape[0] = action_dist.shape[0]`, but found it False"
            )
        if factual_reward.shape[1] != sampled_actions.shape[1]:
            raise ValueError(
                "Expected `factual_reward.shape[1] = action_dist.shape[1]`, but found it False"
            )

        return np.sum(factual_reward * sampled_actions)


def logistic_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    coef_function: coef_func_signature = sample_random_uniform_coefficients,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function for binary rewards.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    coef_function: Callable, default=sample_random_uniform_coefficients
        Function for generating the coefficients used for the context, action and context/action interactions.
        By default, the coefficients are randomly uniformly drawn.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    logits = _base_reward_function(
        context=context,
        action_context=action_context,
        degree=1,
        random_state=random_state,
        coef_function=coef_function,
    )

    return sigmoid(logits)


def logistic_polynomial_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    coef_function: coef_func_signature = sample_random_uniform_coefficients,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function for binary rewards with polynomial feature transformations.

    Note
    ------
    Polynomial and interaction features will be used to calculate the expected rewards.
    Feature transformation is based on `sklearn.preprocessing.PolynomialFeatures(degree=3)`

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    logits = _base_reward_function(
        context=context,
        action_context=action_context,
        degree=3,
        coef_function=coef_function,
        random_state=random_state,
    )

    return sigmoid(logits)


def logistic_sparse_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    coef_function: coef_func_signature = sample_random_uniform_coefficients,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function for binary rewards with small effective feature dimension.

    Note
    ------
    Polynomial and interaction features will be used to calculate the expected rewards.
    `sklearn.preprocessing.PolynomialFeatures(degree=4)` is applied to generate high-dimensional feature vector.
    After that, some dimensions will be dropped as irrelevant dimensions, producing sparse feature vector.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    logits = _base_reward_function(
        context=context,
        action_context=action_context,
        degree=4,
        effective_dim_ratio=0.3,
        coef_function=coef_function,
        random_state=random_state,
    )

    return sigmoid(logits)


def logistic_sparse_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    coef_function: coef_func_signature = sample_random_uniform_coefficients,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function for binary rewards with small effective feature dimension.

    Note
    ------
    Polynomial and interaction features will be used to calculate the expected rewards.
    `sklearn.preprocessing.PolynomialFeatures(degree=4)` is applied to generate high-dimensional feature vector.
    After that, some dimensions will be dropped as irrelevant dimensions, producing sparse feature vector.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    logits = _base_reward_function(
        context=context,
        action_context=action_context,
        degree=4,
        effective_dim_ratio=0.3,
        coef_function=coef_function,
        random_state=random_state,
    )

    return sigmoid(logits)


def linear_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    coef_function: coef_func_signature = sample_random_uniform_coefficients,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear mean reward function for continuous rewards.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return _base_reward_function(
        context=context,
        action_context=action_context,
        degree=1,
        coef_function=coef_function,
        random_state=random_state,
    )


def polynomial_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    coef_function: coef_func_signature = sample_random_uniform_coefficients,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Polynomial mean reward function for continuous rewards.

    Note
    ------
    Polynomial and interaction features will be used to calculate the expected rewards.
    Feature transformation is based on `sklearn.preprocessing.PolynomialFeatures(degree=3)`.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return _base_reward_function(
        context=context,
        action_context=action_context,
        degree=3,
        coef_function=coef_function,
        random_state=random_state,
    )


def sparse_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    coef_function: coef_func_signature = sample_random_uniform_coefficients,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Sparse mean reward function for continuous rewards.

    Note
    ------
    Polynomial and interaction features will be used to calculate the expected rewards.
    `sklearn.preprocessing.PolynomialFeatures(degree=4)` is applied to generate high-dimensional feature vector.
    After that, some dimensions will be dropped as irrelevant dimensions, producing sparse feature vector.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return _base_reward_function(
        context=context,
        action_context=action_context,
        degree=4,
        effective_dim_ratio=0.3,
        coef_function=coef_function,
        random_state=random_state,
    )


def _base_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    degree: int = 3,
    effective_dim_ratio: float = 1.0,
    coef_function: coef_func_signature = sample_random_uniform_coefficients,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Base function to define mean reward functions.

    Note
    ------
    Given context :math:`x` and action_context :math:`a`, this function is used to define
    mean reward function :math:`q(x,a) = \\mathbb{E}[r|x,a]` as follows.

    .. math::

        q(x,a) := \\tilde{x}^T M_{X,A} \\tilde{a} + \\theta_x^T \\tilde{x} + \\theta_a^T \\tilde{a},

    where :math:`x` is a original context vector,
    and :math:`a` is a original action_context vector representing actions.
    Polynomial transformation is applied to original context and action vectors,
    producing :math:`\\tilde{x} \\in \\mathbb{R}^{d_X}` and :math:`\\tilde{a} \\in \\mathbb{R}^{d_A}`.
    Moreover, some dimensions of context and action_context might be randomly dropped according to `effective_dim_ratio`.
    :math:`M_{X,A} \\mathbb{R}^{d_X \\times d_A}`, :math:`\\theta_x \\in \\mathbb{R}^{d_X}`,
    and :math:`\\theta_a \\in \\mathbb{R}^{d_A}` are parameter matrix and vectors,
    all sampled from the uniform distribution.
    The logistic function will be applied to :math:`q(x,a)` in logistic reward functions
    to adjust the range of the function output.

    Currently, this function is used to define
    `obp.dataset.linear_reward function` (degree=1),
    `obp.dataset.polynomial_reward function` (degree=3),
    `obp.dataset.sparse_reward function` (degree=4, effective_dim_ratio=0.1),
     `obp.dataset.logistic_reward function` (degree=1),
     `obp.dataset.logistic_polynomial_reward_function` (degree=3),
     and `obp.dataset.logistic_sparse_reward_function` (degree=4, effective_dim_ratio=0.1).

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    degree: int, default=3
        Specifies the maximal degree of the polynomial feature transformations
        applied to both `context` and `action_context`.

    effective_dim_ratio: int, default=1.0
        Propotion of context dimensions relevant to the expected rewards.
        Specifically, after the polynomial feature transformation is applied to the original context vectors,
        only `dim_context * effective_dim_ratio` fraction of randomly selected dimensions
        will be used as relevant dimensions to generate expected rewards.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    check_scalar(degree, "degree", int, min_val=1)
    check_scalar(
        effective_dim_ratio, "effective_dim_ratio", float, min_val=0, max_val=1
    )
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)

    poly = PolynomialFeatures(degree=degree)
    context_ = poly.fit_transform(context)
    action_context_ = poly.fit_transform(action_context)
    datasize, dim_context = context_.shape
    n_actions, dim_action_context = action_context_.shape
    random_ = check_random_state(random_state)

    if effective_dim_ratio < 1.0:
        effective_dim_context = np.maximum(
            np.int32(dim_context * effective_dim_ratio), 1
        )
        effective_dim_action_context = np.maximum(
            np.int32(dim_action_context * effective_dim_ratio), 1
        )
        effective_context_ = context_[
            :, random_.choice(dim_context, effective_dim_context, replace=False)
        ]
        effective_action_context_ = action_context_[
            :,
            random_.choice(
                dim_action_context, effective_dim_action_context, replace=False
            ),
        ]
    else:
        effective_dim_context = dim_context
        effective_dim_action_context = dim_action_context
        effective_context_ = context_
        effective_action_context_ = action_context_

    context_coef_, action_coef_, context_action_coef_ = coef_function(
        n_rounds=datasize,
        effective_dim_action_context=effective_dim_action_context,
        effective_dim_context=effective_dim_context,
        random_=random_,
    )

    if context_coef_.shape[0] != datasize:
        context_values = np.tile(effective_context_ @ context_coef_, (n_actions, 1)).T
    else:
        context_values = np.tile(
            np.sum(effective_context_ * context_coef_, axis=1), (n_actions, 1)
        ).T

    action_values = action_coef_ @ effective_action_context_.T
    if action_coef_.shape[0] != datasize:
        action_values = np.tile(action_values, (datasize, 1))

    if action_coef_.shape[0] != datasize:
        context_action_values = (
                effective_context_ @ context_action_coef_ @ effective_action_context_.T
        )
    else:
        effective_context_ = np.expand_dims(effective_context_, axis=1)
        context_action_coef_interactions = np.squeeze(np.matmul(effective_context_, context_action_coef_), axis=1)

        context_action_values = (
                context_action_coef_interactions @ effective_action_context_.T
        )

    expected_rewards = context_values + action_values + context_action_values
    # expected_rewards = (
    #     degree * (expected_rewards - expected_rewards.mean()) / expected_rewards.std()
    # )

    return expected_rewards


@dataclass
class ExponentialDelaySampler:
    """Class for sampling delays from different exponential functions.

    Parameters
    -----------
    max_scale: float, default=100.0
        The maximum scale parameter for the exponential delay distribution. When there is no weighted exponential
        function the max_scale becomes the default scale.

    min_scale: float, default=10.0
        The minimum scale parameter for the exponential delay distribution. Only used when sampling from a weighted
        exponential function.

    random_state: int, default=12345
        Controls the random seed in sampling synthetic bandit data.
    """

    max_scale: float = 100.0
    min_scale: float = 10.0
    random_state: int = None

    def __post_init__(self) -> None:
        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)

    def exponential_delay_function(
        self, n_rounds: int, n_actions: int, **kwargs
    ) -> np.ndarray:
        """Exponential delay function used for sampling a number of delay rounds before rewards can be observed.

        Note
        ------
        This implementation of the exponential delay function assumes that there is no causal relationship between the
        context, action or reward and observed delay. Exponential delay function have been observed by Ktena, S.I. et al.

        Parameters
        -----------
        n_rounds: int
            Number of rounds to sample delays for.

        n_actions: int
            Number of actions to sample delays for. If the exponential function is not parameterised the delays are
            repeated for each actions.

        Returns
        ---------
        delay_rounds: array-like, shape (n_rounds, )
            Rounded up round delays representing the amount of rounds before the policy can observe the rewards.

        References
        ------------
        Ktena, S.I., Tejani, A., Theis, L., Myana, P.K., Dilipkumar, D., Huszár, F., Yoo, S. and Shi, W.
        "Addressing delayed feedback for continuous training with neural networks in CTR prediction." 2019.

        """
        delays_per_round = np.ceil(
            self.random_.exponential(scale=self.max_scale, size=n_rounds)
        )

        return np.tile(delays_per_round, (n_actions, 1)).T

    def exponential_delay_function_expected_reward_weighted(
        self, expected_rewards: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Exponential delay function used for sampling a number of delay rounds before rewards can be observed.
        Each delay is conditioned on the expected reward by multiplying (1 - expected_reward) * scale. This creates
        the assumption that the more likely a reward is going be observed, the more likely it will be that the reward
        comes sooner. Eg. recommending an attractive item will likely result in a faster purchase.

         Parameters
         -----------
         expected_rewards : array-like, shape (n_rounds, n_actions)
             The expected reward between 0 and 1 for each arm for each round. This used to weight the scale of the
             exponential function.

         Returns
         ---------
         delay_rounds: array-like, shape (n_rounds, )
             Rounded up round delays representing the amount of rounds before the policy can observe the rewards.
        """
        scale = self.min_scale + (
            (1 - expected_rewards) * (self.max_scale - self.min_scale)
        )
        delays_per_round = np.ceil(
            self.random_.exponential(scale=scale, size=expected_rewards.shape)
        )

        return delays_per_round


def linear_behavior_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear behavior policy function.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    pi_b_logits: array-like, shape (n_rounds, n_actions)
        Logit values given context (:math:`x`).
        The softmax function will be applied to transform it to action choice probabilities.

    """
    return _base_behavior_policy_function(
        context=context,
        action_context=action_context,
        degree=1,
        random_state=random_state,
    )


def polynomial_behavior_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Polynomial behavior policy function.

    Note
    ------
    Polynomial and interaction features will be used to calculate the expected rewards.
    Feature transformation is based on `sklearn.preprocessing.PolynomialFeatures(degree=3)`

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    pi_b_logits: array-like, shape (n_rounds, n_actions)
        Logit values given context (:math:`x`).
        The softmax function will be applied to transform it to action choice probabilities.

    """
    return _base_behavior_policy_function(
        context=context,
        action_context=action_context,
        degree=3,
        random_state=random_state,
    )


def _base_behavior_policy_function(
    context: np.ndarray,
    action_context: np.ndarray,
    degree: int = 3,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Base function to define behavior policy functions.

    Note
    ------
    Given context :math:`x` and action_context :math:`x_a`, this function generates
    logit values for defining a behavior policy as follows.

    .. math::

        f_b(x,a) := \\tilde{x}^T M_{X,A} \\tilde{a} + \\theta_a^T \\tilde{a},

    where :math:`x` is a original context vector,
    and :math:`a` is a original action_context vector representing actions.
    Polynomial transformation is applied to original context and action vectors,
    producing :math:`\\tilde{x} \\in \\mathbb{R}^{d_X}` and :math:`\\tilde{a} \\in \\mathbb{R}^{d_A}`.
    :math:`M_{X,A} \\mathbb{R}^{d_X \\times d_A}` and :math:`\\theta_a \\in \\mathbb{R}^{d_A}` are
    parameter matrix and vector, each sampled from the uniform distribution.
    The softmax function will be applied to :math:`f_b(x,\\cdot)` in `obp.dataset.SyntheticDataset`
    to generate distribution over actions (behavior policy).

    Currently, this function is used to define
    `obp.dataset.linear_behavior_policy` (degree=1)
    and `obp.dataset.polynomial_behavior_policy` (degree=3).

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    degree: int, default=3
        Specifies the maximal degree of the polynomial feature transformations
        applied to both `context` and `action_context`.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    pi_b_logits: array-like, shape (n_rounds, n_actions)
        Logit values given context (:math:`x`).
        The softmax function will be applied to transform it to action choice probabilities.

    """
    check_scalar(degree, "degree", int, min_val=1)
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)

    poly = PolynomialFeatures(degree=degree)
    context_ = poly.fit_transform(context)
    action_context_ = poly.fit_transform(action_context)
    dim_context = context_.shape[1]
    dim_action_context = action_context_.shape[1]

    random_ = check_random_state(random_state)
    action_coef = random_.uniform(size=dim_action_context)
    context_action_coef = random_.uniform(size=(dim_context, dim_action_context))

    pi_b_logits = context_ @ context_action_coef @ action_context_.T
    pi_b_logits += action_coef @ action_context_.T
    pi_b_logits = degree * (pi_b_logits - pi_b_logits.mean()) / pi_b_logits.std()

    return pi_b_logits
