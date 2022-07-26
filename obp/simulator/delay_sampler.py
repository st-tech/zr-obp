from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state


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
