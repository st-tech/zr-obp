# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Regression Model Class for Model-dependent OPE estimators."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, is_classifier


@dataclass
class RegressionModel:
    """ML model to predict the mean reward function (:math:`E[Y | X, A]`).

    Note
    -------
    Reward (or outcome) :math:`Y` must be either binary or continuous.

    Parameters
    ------------
    base_model: BaseEstimator
        Model class to be used to predict the mean reward function.

    fitting_method: str, default='normal'
        Method to fit the regression method.
        Must be one of ['normal', 'iw', 'mrdr'] where 'iw' stands for importance weighting and
        'mrdr' stands for more robust doubly robust.

    References
    -----------
    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    """

    base_model: BaseEstimator
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert self.fitting_method in [
            "normal",
            "iw",
            "mrdr",
        ], f"fitting method must be one of 'normal', 'iw', or 'mrdr', but {self.fitting_method} is given"

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        action_context: np.ndarray,
    ) -> None:
        """Fit the regression model on given logged bandit feedback data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in the given training logged bandit feedback.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        reward: array-like, shape (n_rounds,)
            Observed rewards in the given training logged bandit feedback.

        pscore: Optional[np.ndarray], default: None
            Propensity scores, the probability of selecting each action by behavior policy,
            in the given training logged bandit feedback.

        action_context: array-like, shape (n_actions, dim_action_context)
            Context vector characterizing each action.

        """
        # create context vector to make predictions
        X = self._pre_process_for_reg_model(
            context=context, action=action, action_context=action_context,
        )
        # train the base model according to the given `fitting method`
        if self.fitting_method == "normal":
            self.base_model.fit(X, reward)
        elif self.fitting_method == "iw":
            sample_weight = np.mean(pscore) / pscore
            self.base_model.fit(X, reward, sample_weight=sample_weight)
        elif self.fitting_method == "mrdr":
            sample_weight = (1.0 - pscore) / pscore ** 2
            self.base_model.fit(X, reward, sample_weight=sample_weight)

    def predict(
        self,
        context: np.ndarray,
        action_context: np.ndarray,
        selected_actions: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict the mean reward function.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in the given training logged bandit feedback.

        action_context: array-like, shape shape (n_actions, dim_action_context)
            Context vector characterizing each action.

        selected_actions: array-like, shape (n_rounds, len_list)
            Sequence of actions selected by counterfactual (or evaluation) policy
            at each round in offline bandit simulation.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.

        Returns
        -----------
        estimated_rewards: array-like, shape (n_rounds, )
            Estimated rewards by regression model for each round.

        """
        # create context vector to make predictions
        selected_actions_at_positions = selected_actions[
            np.arange(position.shape[0]), position
        ]
        X = self._pre_process_for_reg_model(
            context=context,
            action=selected_actions_at_positions,
            action_context=action_context,
        )
        # make predictions
        if is_classifier(self.base_model):
            return self.base_model.predict_proba(X)[:, 1]
        else:
            return self.base_model.predict(X)

    def _pre_process_for_reg_model(
        self, context: np.ndarray, action: np.ndarray, action_context: np.ndarray,
    ) -> np.ndarray:
        """Preprocess feature vectors to train a give regression model.

        Note
        -----
        Please override this method if you want to use another feature enginnering for training the regression model.

        Parameters
        -----------
        context: array-like, shape (n_rounds,)
            Context vectors in the given training logged bandit feedback.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        action_context: array-like, shape shape (n_actions, dim_action_context)
            Context vector characterizing each action.

        """
        return np.c_[context, action_context[action]]
