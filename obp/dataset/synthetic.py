# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic Logged Bandit Feedback."""
from dataclasses import dataclass
from typing import Callable
from typing import Optional

import numpy as np
from scipy.stats import truncnorm
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from ..types import BanditFeedback
from ..utils import check_array
from ..utils import sample_action_fast
from ..utils import sigmoid
from ..utils import softmax
from .base import BaseBanditDataset
from .reward_type import RewardType


@dataclass
class SyntheticBanditDataset(BaseBanditDataset):
    """Class for generating synthetic bandit dataset.

    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times,
    we have different bandit samples with the same setting.
    This can be used to estimate confidence intervals of the performances of OPE estimators.

    If None is set as `behavior_policy_function`, the synthetic data will be context-free bandit feedback.

    Parameters
    -----------
    n_actions: int
        Number of actions.

    dim_context: int, default=1
        Number of dimensions of context vectors.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary' is given, rewards are sampled from the Bernoulli distribution.
        When 'continuous' is given, rewards are sampled from the truncated Normal distribution with `scale=1`.
        The mean parameter of the reward distribution is determined by the `reward_function` specified by the next argument.

    reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function generating expected reward for each given action-context pair,
        i.e., :math:`\\mu: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None is set, context **independent** expected reward for each action will be
        sampled from the uniform distribution automatically.

    reward_std: float, default=1.0
        Standard deviation of the reward distribution.
        A larger value leads to a noisy reward distribution.
        This argument is valid only when `reward_type="continuous"`.

    behavior_policy_function: Callable[[np.ndarray, np.ndarray], np.ndarray], default=None
        Function generating probability distribution over action space,
        i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.
        If None is set, context **independent** uniform distribution will be used (uniform random behavior policy).

    tau: float, default=1.0
        A temperature hyperparameer which controls the behavior policy.
        A large value leads to a near-uniform behavior policy,
        while a small value leads to a near-deterministic behavior policy.

    random_state: int, default=12345
        Controls the random seed in sampling synthetic bandit dataset.

    dataset_name: str, default='synthetic_bandit_dataset'
        Name of the dataset.

    Examples
    ----------

    .. code-block:: python

        >>> import numpy as np
        >>> from obp.dataset import (
            SyntheticBanditDataset,
            linear_reward_function,
            linear_behavior_policy
        )

        # generate synthetic contextual bandit feedback with 10 actions.
        >>> dataset = SyntheticBanditDataset(
                n_actions=10,
                dim_context=5,
                reward_function=logistic_reward_function,
                behavior_policy=linear_behavior_policy,
                random_state=12345
            )
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=100000)
        >>> bandit_feedback
        {
            'n_rounds': 100000,
            'n_actions': 10,
            'context': array([[-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057],
                    [ 1.39340583,  0.09290788,  0.28174615,  0.76902257,  1.24643474],
                    [ 1.00718936, -1.29622111,  0.27499163,  0.22891288,  1.35291684],
                    ...,
                    [ 1.36946256,  0.58727761, -0.69296769, -0.27519988, -2.10289159],
                    [-0.27428715,  0.52635353,  1.02572168, -0.18486381,  0.72464834],
                    [-1.25579833, -1.42455203, -0.26361242,  0.27928604,  1.21015571]]),
            'action_context': array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]),
            'action': array([7, 4, 0, ..., 7, 9, 6]),
            'position': None,
            'reward': array([0, 1, 1, ..., 0, 1, 0]),
            'expected_reward': array([[0.80210203, 0.73828559, 0.83199558, ..., 0.81190503, 0.70617705,
                    0.68985306],
                    [0.94119582, 0.93473317, 0.91345213, ..., 0.94140688, 0.93152449,
                    0.90132868],
                    [0.87248862, 0.67974991, 0.66965669, ..., 0.79229752, 0.82712978,
                    0.74923536],
                    ...,
                    [0.64856003, 0.38145901, 0.84476094, ..., 0.40962057, 0.77114661,
                    0.65752798],
                    [0.73208527, 0.82012699, 0.78161352, ..., 0.72361416, 0.8652249 ,
                    0.82571751],
                    [0.40348366, 0.24485417, 0.24037926, ..., 0.49613133, 0.30714854,
                    0.5527749 ]]),
            'pscore': array([0.05423855, 0.10339675, 0.09756788, ..., 0.05423855, 0.07250876,
                    0.14065505])
        }

    """

    n_actions: int
    dim_context: int = 1
    reward_type: str = RewardType.BINARY.value
    reward_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    reward_std: float = 1.0
    behavior_policy_function: Optional[
        Callable[[np.ndarray, np.ndarray], np.ndarray]
    ] = None
    tau: float = 1.0
    random_state: int = 12345
    dataset_name: str = "synthetic_bandit_dataset"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.dim_context, "dim_context", int, min_val=1)
        if RewardType(self.reward_type) not in [
            RewardType.BINARY,
            RewardType.CONTINUOUS,
        ]:
            raise ValueError(
                f"reward_type must be either '{RewardType.BINARY.value}' or '{RewardType.CONTINUOUS.value}', but {self.reward_type} is given.'"
            )
        check_scalar(self.reward_std, "reward_std", (int, float), min_val=0)
        check_scalar(self.tau, "tau", (int, float), min_val=0)
        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)
        if self.reward_function is None:
            self.expected_reward = self.sample_contextfree_expected_reward()
        if self.behavior_policy_function is None:
            self.behavior_policy = np.ones(self.n_actions) / self.n_actions
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            self.reward_min = 0
            self.reward_max = 1e10
        # one-hot encoding representations characterizing each action
        self.action_context = np.eye(self.n_actions, dtype=int)

    @property
    def len_list(self) -> int:
        """Length of recommendation lists."""
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
            )

        return expected_reward_

    def sample_reward_given_expected_reward(
        self,
        expected_reward: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Sample reward given expected rewards"""
        expected_reward_factual = expected_reward[np.arange(action.shape[0]), action]
        if RewardType(self.reward_type) == RewardType.BINARY:
            reward = self.random_.binomial(n=1, p=expected_reward_factual)
        elif RewardType(self.reward_type) == RewardType.CONTINUOUS:
            mean = expected_reward_factual
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
        """Sample rewards given contexts and actions, i.e., :math:`r \\sim p(r \\mid x, a)`.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors characterizing each round (such as user information).

        action: array-like, shape (n_rounds,)
            Selected actions to the contexts.

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
        """Obtain batch logged bandit feedback.

        Parameters
        ----------
        n_rounds: int
            Number of rounds for synthetic bandit feedback data.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Generated synthetic bandit feedback dataset.

        """
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        context = self.random_.normal(size=(n_rounds, self.dim_context))
        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is None:
            behavior_policy_ = np.tile(self.behavior_policy, (n_rounds, 1))
            behavior_policy_ = softmax(behavior_policy_ / self.tau)
            action = self.random_.choice(
                np.arange(self.n_actions), p=self.behavior_policy, size=n_rounds
            )
        else:
            behavior_policy_ = self.behavior_policy_function(
                context=context,
                action_context=self.action_context,
                random_state=self.random_state,
            )
            behavior_policy_ = softmax(behavior_policy_ / self.tau)
            action = sample_action_fast(
                behavior_policy_, random_state=self.random_state
            )
        pscore = behavior_policy_[np.arange(n_rounds), action]

        # sample reward based on the context and action
        expected_reward_ = self.calc_expected_reward(context)
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            # correct expected_reward_, as we use truncated normal distribution here
            mean = expected_reward_
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            expected_reward_ = truncnorm.stats(
                a=a, b=b, loc=mean, scale=self.reward_std, moments="m"
            )
        reward = self.sample_reward_given_expected_reward(expected_reward_, action)

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=context,
            action_context=self.action_context,
            action=action,
            position=None,  # position effect is not considered in synthetic data
            reward=reward,
            expected_reward=expected_reward_,
            pscore=pscore,
        )

    def calc_ground_truth_policy_value(
        self, expected_reward: np.ndarray, action_dist: np.ndarray
    ) -> float:
        """Calculate the policy value of given action distribution on the given expected_reward.

        Parameters
        -----------
        expected_reward: array-like, shape (n_rounds, n_actions)
            Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.
            This is often the expected_reward of the test set of logged bandit feedback data.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        Returns
        ----------
        policy_value: float
            The policy value of the given action distribution on the given bandit feedback data.

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


def logistic_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Logistic mean reward function for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    # each arm has different coefficient vectors
    coef_ = random_.uniform(size=(action_context.shape[0], context.shape[1]))
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        logits[:, d] = context @ coef_[d] + action_context[d] @ action_coef_

    return sigmoid(logits)


def linear_reward_function(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear mean reward function for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_reward: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`), i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)

    random_ = check_random_state(random_state)
    expected_reward = np.zeros((context.shape[0], action_context.shape[0]))
    # each arm has different coefficient vectors
    coef_ = random_.uniform(size=(action_context.shape[0], context.shape[1]))
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        expected_reward[:, d] = context @ coef_[d] + action_context[d] @ action_coef_

    return expected_reward


def linear_behavior_policy(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Linear contextual behavior policy for synthetic bandit datasets.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each round (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation for each action.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    behavior_policy: array-like, shape (n_rounds, n_actions)
        Logit values given context (:math:`x`), i.e., :math:`\\pi: \\mathcal{X} \\rightarrow \\Delta(\\mathcal{A})`.

    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action_context, name="action_context", expected_dim=2)

    random_ = check_random_state(random_state)
    logits = np.zeros((context.shape[0], action_context.shape[0]))
    coef_ = random_.uniform(size=context.shape[1])
    action_coef_ = random_.uniform(size=action_context.shape[1])
    for d in np.arange(action_context.shape[0]):
        logits[:, d] = context @ coef_ + action_context[d] @ action_coef_

    return logits
