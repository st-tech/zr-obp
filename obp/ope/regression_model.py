# Copyright (c) ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator

from ..dataset import LogBanditFeedback


@dataclass
class RegressionModel:
    """ML model to predict the mean reward function (:math:`E[Y | X, A]`).

    Parameters
    ----------
    base_model: BaseEstimator
        Model class to be used to predict the mean reward function.

    fitting_method: str, default='normal'
        Method to fit the regression method.
        Choose among ['normal', 'iw', 'mrdr'] where 'iw' stands for importance weighting and
        'mrdr' stands for more robust doubly robust.

    is_predict_proba: bool, default: True
        Whether the regression model predicts probabilities (classification) or not (regression).

    References
    -----------
    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    """
    base_model: BaseEstimator
    fitting_method: str = 'normal'
    # n_folds: int = 1  # TODO: implement crossfittig
    is_predict_proba: bool = True

    def fit(self, train: LogBanditFeedback, action_context: np.ndarray) -> None:
        """Fit the regression model on given logged bandit feedback data.

        Parameters
        ----------
        train: LogBanditFeedback
            Training set of logged bandit feedback data to be used in offline bandit simulation.

        action_context: array-like, shape (n_actions, dim_action_context)
            Context vector characterizing each action.

        """
        X = self._pre_process_for_reg_model(
            train=train, action_context=action_context, action=train['action'])
        if self.fitting_method == 'normal':
            self.base_model.fit(X, train['reward'])
        elif self.fitting_method == 'iw':
            sample_weight = np.mean(train['pscore']) / train['pscore']
            self.base_model.fit(X, train['reward'], sample_weight=sample_weight)
        elif self.fitting_method == 'mrdr':
            sample_weight = ((1. - train['pscore']) / train['pscore']**2)
            self.base_model.fit(X, train['reward'], sample_weight=sample_weight)
        else:
            raise ValueError(f"Undefined fitting_method {self.fitting_method} is given.")

    def predict(self,
                train: LogBanditFeedback,
                action_context: np.ndarray,
                selected_actions: np.ndarray) -> np.ndarray:
        """Predict the mean reward function.

        Parameters
        -----------
        train: LogBanditFeedback
            Training set of logged bandit feedback data to be used in offline bandit simulation.

        action_context: array-like, shape shape (n_actions, dim_action_context)
            Context vector characterizing each action.

        selected_actions: array-like, shape (n_rounds, len_list)
            Lists of actions selected by counterfactual (or evaluation) policy at each round in offline bandit simulation.

        Returns
        -----------
        estimated_rewards: array-like, shape (n_rounds, )
            Estimated rewards by regression model for each round.

        """
        selected_actions_at_positions = selected_actions[np.arange(train['n_rounds']), train['position']]
        X = self._pre_process_for_reg_model(
            train=train, action_context=action_context, action=selected_actions_at_positions)
        if self.is_predict_proba:
            return self.base_model.predict_proba(X)[:, 1]
        else:
            return self.base_model.predict(X)

    def _pre_process_for_reg_model(self,
                                   train: LogBanditFeedback,
                                   action_context: np.ndarray,
                                   action: np.ndarray) -> np.ndarray:
        """Preprocess feature vectors to train a give regression model.

        Note
        -----
        Please override this method if you want to use another feature enginnering for training the regression model.

        Parameters
        -----------
        train: LogBanditFeedback
            Training set of logged bandit feedback data to be used in offline bandit simulation.

        action_context: array-like, shape shape (n_actions, dim_action_context)
            Context vector characterizing each action.

        action: array-like, shape (n_rounds, )
            Actions for each round.

        """
        return np.c_[train['position'], train['context'], action_context[action]]
