from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from sklearn.utils import check_random_state

from obp.dataset.synthetic import sample_random_uniform_coefficients


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
    drift_interval: int
        Controls interval of steps at which the coefficients are updated.

    transition_period: int, default=0
        Controls the period in which the coefficients are transitioning between new and old. The transition period
        always happened before the drift interval. Meaning, that if the drift interval is 5000 and the transition period
        500, the transition will be between step 4500 and step 5000.

    transition_type: str, default="linear"
        The type of transition (linear or weighted_sampled) to be applied while transitioning between two sets of
        coefficients.

    seasonal: bool, default=False
        When True, the coefficients will shift between two sets of coefficients representing a seasonal shift.

    base_coefficient_weight: float, default=0.0
        A floating point between 0.0 and 1.0 that represents a base coefficient weight that is consistent regardless of
        any drift. This can be used to ensure the severity of the drift over time.

    effective_dim_action_context: (optional) int, default=None
        Specifies the dimensions of the action context coefficients.

    effective_dim_context: (optional) int, default=None
        Specifies the dimensions of the context coefficients.

    random_state: int, default=12345
        Controls the random seed

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
            (
                self.base_context_coef,
                self.base_action_coef,
                self.base_context_action_coef,
            ) = sample_random_uniform_coefficients(
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
                self.append_current_coefs(
                    context_coefs,
                    action_coefs,
                    context_action_coefs,
                    rounds=self.available_rounds,
                )
                required_rounds -= self.available_rounds
                self.update_coef()
                self.available_rounds = self.drift_interval
            else:
                self.append_current_coefs(
                    context_coefs,
                    action_coefs,
                    context_action_coefs,
                    rounds=required_rounds,
                )
                self.available_rounds -= required_rounds
                required_rounds = 0

        return (
            np.vstack(context_coefs),
            np.vstack(action_coefs),
            np.vstack(context_action_coefs),
        )

    def append_current_coefs(
        self,
        context_coefs: List[np.ndarray],
        action_coefs: List[np.ndarray],
        context_action_coefs: List[np.ndarray],
        rounds: int,
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

        if self.transition_type == "weighted_sampled":
            weights = self.random_.binomial(n=1, p=weights)

        context_coefs.append(
            self.compute_weighted_coefs(
                self.context_coefs, self.base_context_coef, rounds, weights
            )
        )
        action_coefs.append(
            self.compute_weighted_coefs(
                self.action_coefs, self.base_action_coef, rounds, weights
            )
        )
        context_action_coefs.append(
            self.compute_weighted_coefs(
                self.context_action_coefs,
                self.base_context_action_coef,
                rounds,
                weights,
            )
        )

    def compute_weighted_coefs(self, coefs, base_coef, rounds, weights):
        base_coef = self.base_coefficient_weight * base_coef

        A = np.tile(coefs[0], [rounds] + [1 for _ in coefs[0].shape])
        B = np.tile(coefs[1], [rounds] + [1 for _ in coefs[1].shape])
        coefs = (
            base_coef
            + A
            * np.expand_dims(
                (1 - self.base_coefficient_weight) * (1 - weights),
                list(range(1, len(A.shape))),
            )
            + B
            * np.expand_dims(
                (1 - self.base_coefficient_weight) * weights,
                list(range(1, len(B.shape))),
            )
        )
        return coefs
