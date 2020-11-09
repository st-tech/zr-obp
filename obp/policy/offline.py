# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Offline Bandit Algorithms."""
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from sklearn.base import clone, ClassifierMixin, is_classifier
from sklearn.linear_model import LogisticRegression

from .base import BaseOffPolicyLearner
from ..utils import check_bandit_feedback_inputs


@dataclass
class IPWLearner(BaseOffPolicyLearner):
    """Off-policy learner with Inverse Probability Weighting.

    Parameters
    -----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of recommended actions in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    base_classifier: ClassifierMixin
        Machine learning classifier used to train an offline decision making policy.

    References
    ------------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    base_classifier: Optional[ClassifierMixin] = None

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        if self.base_classifier is None:
            self.base_classifier = LogisticRegression(random_state=12345)
        else:
            assert is_classifier(
                self.base_classifier
            ), "base_classifier must be a classifier"
        self.base_classifier_list = [
            clone(self.base_classifier) for _ in np.arange(self.len_list)
        ]

    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create training data for off-policy learning.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Propensity scores, the probability of selecting each action by behavior policy,
            in the given logged bandit feedback.

        Returns
        --------
        (X, sample_weight, y): Tuple[np.ndarray, np.ndarray, np.ndarray]
            Feature vectors, sample weights, and outcome for training the base machine learning model.

        """
        return context, (reward / pscore), action

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> None:
        """Fits the offline bandit policy according to the given logged bandit feedback data.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Sampled (realized) actions by behavior policy in each round, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Propensity scores or the action choice probabilities by behavior policy, i.e., :math:`\\pi_b(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.
            If None is given, a learner assumes that there is only one position.
            When `len_list` > 1, position has to be set.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )
        if pscore is None:
            n_actions = np.int(action.max() + 1)
            pscore = np.ones_like(action) / n_actions
        if self.len_list == 1:
            position = np.zeros_like(action, dtype=int)
        else:
            assert (
                position is not None
            ), "position has to be set when len_list is larger than 1"

        for position_ in np.arange(self.len_list):
            X, sample_weight, y = self._create_train_data_for_opl(
                context=context[position == position_],
                action=action[position == position_],
                reward=reward[position == position_],
                pscore=pscore[position == position_],
            )
            self.base_classifier_list[position_].fit(
                X=X, y=y, sample_weight=sample_weight
            )

    def predict(self, context: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """Predict best action for new data.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        epsilon: float, default=0.0
            Exploration hyperparameter that must take value in the interval [0.0, 1.0].
            A positive value of epsilon makes the policy stochastic, making sure that the
            overlap condition is satisfied in the resulting logged bandit feedback.

        Returns
        -----------
        action_dist: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choice probabilities by a trained classifier.

        """
        assert (
            isinstance(context, np.ndarray) and context.ndim == 2
        ), "context must be 2-dimensional ndarray"
        assert (0.0 <= epsilon <= 1.0) and isinstance(
            epsilon, float
        ), f"epsilon must be a float in the interval [0.0, 1.0], but {epsilon} is given"

        n_rounds = context.shape[0]
        action_dist = np.ones((n_rounds, self.n_actions, self.len_list))
        action_dist *= epsilon * (1.0 / self.n_actions)
        for position_ in np.arange(self.len_list):
            predicted_actions_at_position = self.base_classifier_list[
                position_
            ].predict(context)
            action_dist[
                np.arange(n_rounds),
                predicted_actions_at_position,
                np.ones(n_rounds, dtype=int) * position_,
            ] += (1 - epsilon)
        return action_dist
