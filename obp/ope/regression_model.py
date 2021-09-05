# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Regression Model Class for Estimating Mean Reward Functions."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from ..utils import check_bandit_feedback_inputs


@dataclass
class RegressionModel(BaseEstimator):
    """Machine learning model to estimate the mean reward function (:math:`q(x,a):= \\mathbb{E}[r|x,a]`).

    Note
    -------
    Reward (or outcome) :math:`r` must be either binary or continuous.

    Parameters
    ------------
    base_model: BaseEstimator
        A machine learning model used to estimate the mean reward function.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    action_context: array-like, shape (n_actions, dim_action_context), default=None
        Context vector characterizing action (i.e., vector representation of each action).
        If not given, one-hot encoding of the action variable is used as default.

    fitting_method: str, default='normal'
        Method to fit the regression model.
        Must be one of ['normal', 'iw', 'mrdr'] where 'iw' stands for importance weighting and
        'mrdr' stands for more robust doubly robust.

    References
    -----------
    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    Yusuke Narita, Shota Yasui, and Kohei Yata.
    "Off-policy Bandit and Reinforcement Learning.", 2020.

    """

    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["normal", "iw", "mrdr"]
        ):
            raise ValueError(
                f"fitting_method must be one of 'normal', 'iw', or 'mrdr', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "base_model must be BaseEstimator or a child class of BaseEstimator"
            )

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
        action_dist: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the regression model on given logged bandit feedback data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.
            When None is given, behavior policy is assumed to be uniform.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is set, a regression model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.

        action_dist: array-like, shape (n_rounds, n_actions, len_list), default=None
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
            When either of 'iw' or 'mrdr' is used as the 'fitting_method' argument, then `action_dist` must be given.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            action_context=self.action_context,
        )
        n_rounds = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"position elements must be smaller than len_list, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when fitting_method is either 'iw' or 'mrdr', action_dist (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of action_dist must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
            if not np.allclose(action_dist.sum(axis=1), 1):
                raise ValueError("action_dist must be a probability distribution")
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        for position_ in np.arange(self.len_list):
            idx = position == position_
            X = self._pre_process_for_reg_model(
                context=context[idx],
                action=action[idx],
                action_context=self.action_context,
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {position_}")
            # train the base model according to the given `fitting method`
            if self.fitting_method == "normal":
                self.base_model_list[position_].fit(X, reward[idx])
            else:
                action_dist_at_position = action_dist[
                    np.arange(n_rounds),
                    action,
                    position_ * np.ones(n_rounds, dtype=int),
                ][idx]
                if self.fitting_method == "iw":
                    sample_weight = action_dist_at_position / pscore[idx]
                    self.base_model_list[position_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )
                elif self.fitting_method == "mrdr":
                    sample_weight = action_dist_at_position
                    sample_weight *= 1.0 - pscore[idx]
                    sample_weight /= pscore[idx] ** 2
                    self.base_model_list[position_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict the mean reward function.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors of new data.

        Returns
        -----------
        estimated_rewards_by_reg_model: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Expected rewards of new data estimated by the regression model.

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

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
        n_folds: int = 1,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Fit the regression model on given logged bandit feedback data and predict the reward function of the same data.

        Note
        ------
        When `n_folds` is larger than 1, then the cross-fitting procedure is applied.
        See the reference for the details about the cross-fitting technique.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities (propensity score) of a behavior policy
            in the training logged bandit feedback.
            When None is given, the the behavior policy is assumed to be a uniform one.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is set, a regression model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.

        action_dist: array-like, shape (n_rounds, n_actions, len_list), default=None
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
            When either of 'iw' or 'mrdr' is used as the 'fitting_method' argument, then `action_dist` must be given.

        n_folds: int, default=1
            Number of folds in the cross-fitting procedure.
            When 1 is given, the regression model is trained on the whole logged bandit feedback data.
            Please refer to https://arxiv.org/abs/2002.08536 about the details of the cross-fitting procedure.

        random_state: int, default=None
            `random_state` affects the ordering of the indices, which controls the randomness of each fold.
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html for the details.

        Returns
        -----------
        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards of new data estimated by the regression model.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            action_context=self.action_context,
        )
        n_rounds = context.shape[0]

        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"position elements must be smaller than len_list, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when fitting_method is either 'iw' or 'mrdr', action_dist (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of action_dist must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(
                context=context,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                action_dist=action_dist,
            )
            return self.predict(context=context)
        else:
            estimated_rewards_by_reg_model = np.zeros(
                (n_rounds, self.n_actions, self.len_list)
            )
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        for train_idx, test_idx in kf.split(context):
            action_dist_tr = (
                action_dist[train_idx] if action_dist is not None else action_dist
            )
            self.fit(
                context=context[train_idx],
                action=action[train_idx],
                reward=reward[train_idx],
                pscore=pscore[train_idx],
                position=position[train_idx],
                action_dist=action_dist_tr,
            )
            estimated_rewards_by_reg_model[test_idx, :, :] = self.predict(
                context=context[test_idx]
            )
        return estimated_rewards_by_reg_model

    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
    ) -> np.ndarray:
        """Preprocess feature vectors to train a regression model.

        Note
        -----
        Please override this method if you want to use another feature enginnering
        for training the regression model.

        Parameters
        -----------
        context: array-like, shape (n_rounds,)
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        action_context: array-like, shape shape (n_actions, dim_action_context)
            Context vector characterizing action (i.e., vector representation of each action).

        """
        return np.c_[context, action_context[action]]
