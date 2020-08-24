# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Offline Bandit Algorithms."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .base import BaseOffPolicyLearner


@dataclass
class IPWLearner(BaseOffPolicyLearner):
    """Off-policy learner with Inverse Probability Weighting.

    Parameters
    -----------
    base_model: ClassifierMixin
        Machine learning classifier to be used to create the decision making policy.

    References
    ------------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    """

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

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
            Context vectors in the given training logged bandit feedback.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        reward: array-like, shape (n_rounds,)
            Observed rewards in the given training logged bandit feedback.

        pscore: array-like, shape (n_rounds,), default: None
            Propensity scores, the probability of selecting each action by behavior policy,
            in the given training logged bandit feedback.

        Returns
        --------
        (X, sample_weight, y): Tuple[np.ndarray, np.ndarray, np.ndarray]
            Feature vectors, sample weights, and outcome for training the base machine learning model.

        """
        return context, (reward / pscore), action
