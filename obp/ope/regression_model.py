# Copyright (c) ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Regression Model Class for Model-dependent OPE estimators."""
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, is_classifier

from ..dataset import BanditFeedback


@dataclass
class RegressionModel:
    """ML model to predict the mean reward function (:math:`E[Y | X, A]`).

    Parameters
    ----------
    base_model: BaseEstimator
        Model class to be used to predict the mean reward function.

    fitting_method: str, default='normal'
        Method to fit the regression method.
        Must choose among ['normal', 'iw', 'mrdr'] where 'iw' stands for importance weighting and
        'mrdr' stands for more robust doubly robust.

    Note
    ------
    Reward (or outcome) :math:`Y` must be either binary or continuous.

    References
    -----------
    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    """
    base_model: BaseEstimator
    fitting_method: str = 'normal'

    def fit(self, bandit_feedback: BanditFeedback, action_context: np.ndarray) -> None:
        """Fit the regression model on given logged bandit feedback data.

        Parameters
        ----------
        bandit_feedback: BanditFeedback
            Logged bandit feedback data to be used in offline bandit simulation.

        action_context: array-like, shape (n_actions, dim_action_context)
            Context vector characterizing each action.

        """
        # create context vector to make predictions
        X = self._pre_process_for_reg_model(
            bandit_feedback=bandit_feedback,
            action_context=action_context,
            action=bandit_feedback['action']
        )
        # train the base model according to the given `fitting method`
        if self.fitting_method == 'normal':
            self.base_model.fit(X, bandit_feedback['reward'])
        elif self.fitting_method == 'iw':
            sample_weight = np.mean(bandit_feedback['pscore']) / bandit_feedback['pscore']
            self.base_model.fit(X, bandit_feedback['reward'], sample_weight=sample_weight)
        elif self.fitting_method == 'mrdr':
            sample_weight = ((1. - bandit_feedback['pscore']) / bandit_feedback['pscore']**2)
            self.base_model.fit(X, bandit_feedback['reward'], sample_weight=sample_weight)
        else:
            raise ValueError(f"Undefined fitting_method {self.fitting_method} is given.")

    def predict(self,
                bandit_feedback: BanditFeedback,
                action_context: np.ndarray,
                selected_actions: np.ndarray) -> np.ndarray:
        """Predict the mean reward function.

        Parameters
        -----------
        bandit_feedback: BanditFeedback
            Logged bandit feedback data to be used in offline bandit simulation.

        action_context: array-like, shape shape (n_actions, dim_action_context)
            Context vector characterizing each action.

        selected_actions: array-like, shape (n_rounds, len_list)
            Lists of actions selected by counterfactual (or evaluation) policy at each round in offline bandit simulation.

        Returns
        -----------
        estimated_rewards: array-like, shape (n_rounds, )
            Estimated rewards by regression model for each round.

        """
        # create context vector to make predictions
        selected_actions_at_positions = selected_actions[
            np.arange(bandit_feedback['n_rounds']),
            bandit_feedback['position']
        ]
        X = self._pre_process_for_reg_model(
            bandit_feedback=bandit_feedback,
            action_context=action_context,
            action=selected_actions_at_positions
        )
        # make predictions
        if is_classifier(self.base_model):
            return self.base_model.predict_proba(X)[:, 1]
        else:
            return self.base_model.predict(X)

    def _pre_process_for_reg_model(self,
                                   bandit_feedback: BanditFeedback,
                                   action_context: np.ndarray,
                                   action: np.ndarray) -> np.ndarray:
        """Preprocess feature vectors to train a give regression model.

        Note
        -----
        Please override this method if you want to use another feature enginnering for training the regression model.

        Parameters
        -----------
        bandit_feedback: BanditFeedback
            Logged bandit feedback data to be used in offline bandit simulation.

        action_context: array-like, shape shape (n_actions, dim_action_context)
            Context vector characterizing each action.

        action: array-like, shape (n_rounds, )
            Actions for each round.

        """
        return np.c_[bandit_feedback['position'], bandit_feedback['context'], action_context[action]]
