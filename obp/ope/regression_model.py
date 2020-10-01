# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Regression Model Class for Model-dependent OPE estimators."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier

from ..utils import check_bandit_feedback_inputs


@dataclass
class RegressionModel:
    """ML model to estimate the mean reward function (:math:`\\mu(x, a) = \\mathbbb{E} [Y(a) | X=x]`).

    Note
    -------
    Reward (or outcome) :math:`Y` must be either binary or continuous.

    Parameters
    ------------
    base_model: BaseEstimator
        Model class to be used to predict the mean reward function.

    n_actions: int
        Number of actions.

    len_list: int, default: 1
        Length of a list of recommended actions in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    action_context: array-like, shape (n_actions, dim_action_context), default=None
        Context vector characterizing each action.
        If not given, then one-hot encoding of the action variable is automatically used as `action_context`.

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
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize Class."""
        assert self.fitting_method in [
            "normal",
            "iw",
            "mrdr",
        ], f"fitting method must be one of 'normal', 'iw', or 'mrdr', but {self.fitting_method} is given"
        assert self.n_actions > 1 and isinstance(
            self.n_actions, int
        ), f"n_actions must be an integer larger than 1, but {self.n_actions} is given"
        assert self.len_list > 0 and isinstance(
            self.len_list, int
        ), f"len_list must be a positive integer, but {self.len_list} is given"
        self.base_model_list = [
            clone(self.base_model) for _ in np.arange(self.len_list)
        ]
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
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

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given training logged bandit feedback.
            If None is given, a learner assumes that there is only one position.
            When `len_list` > 1, position has to be set.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            action_context=self.action_context,
        )
        if position is None:
            assert self.len_list == 1, "position has to be set when len_list is 1"
            position = np.zeros_like(action)
        for position_ in np.arange(self.len_list):
            idx = position == position_
            X = self._pre_process_for_reg_model(
                context=context[idx],
                action=action[idx],
                action_context=self.action_context,
            )
            # train the base model according to the given `fitting method`
            if self.fitting_method == "normal":
                self.base_model_list[position_].fit(X, reward[idx])
            elif self.fitting_method == "iw":
                sample_weight = 1.0 / pscore[idx]
                self.base_model_list[position_].fit(
                    X, reward[idx], sample_weight=sample_weight
                )
            elif self.fitting_method == "mrdr":
                sample_weight = (1.0 - pscore[idx]) / (pscore[idx] ** 2)
                self.base_model_list[position_].fit(
                    X, reward[idx], sample_weight=sample_weight
                )

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict the mean reward function.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        estimated_rewards_by_reg_model: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Estimated expected rewards for new data given each item and position by the regression model.

        """
        n_rounds_of_new_data = context.shape[0]
        ones_n_rounds_arr = np.ones(n_rounds_of_new_data, int)
        estimated_rewards_by_reg_model = np.zeros(
            (n_rounds_of_new_data, self.n_actions, self.len_list)
        )
        for action_ in np.arange(self.n_actions):
            for position_ in np.arange(self.len_list):
                X = self._pre_process_for_reg_model(
                    context=context,
                    action=action_ * ones_n_rounds_arr,
                    action_context=self.action_context,
                )
                estimated_rewards_ = (
                    self.base_model_list[position_].predict_proba(X)[:, 1]
                    if is_classifier(self.base_model_list[position_])
                    else self.base_model_list[position_].predict(X)
                )
                estimated_rewards_by_reg_model[
                    np.arange(n_rounds_of_new_data),
                    action_ * ones_n_rounds_arr,
                    position_ * ones_n_rounds_arr,
                ] = estimated_rewards_
        return estimated_rewards_by_reg_model

    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Preprocess feature vectors to train a give regression model.

        Note
        -----
        Please override this method if you want to use another feature enginnering
        for training the regression model.

        Parameters
        -----------
        context: array-like, shape (n_rounds,)
            Context vectors in the given training logged bandit feedback.

        action: array-like, shape (n_rounds,)
            Selected actions by behavior policy in the given training logged bandit feedback.

        action_context: array-like, shape shape (n_actions, dim_action_context), default=None
            Context vector characterizing each action.

        """
        return np.c_[context, action_context[action]]
