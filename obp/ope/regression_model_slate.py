# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Regression Model Class for Estimating Baseline Values in Cascade-DR."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar


@dataclass
class SlateRegressionModel(BaseEstimator):
    """Machine learning model to estimate the baseline values
        (:math:`\\hat{Q}_k(x, a(1), \\ldots, a(k)) \\approx \\mathbb{E}[ \sum_{k'=k}^K \\alpha_{k'} r(k') | x, a(1), \\ldots, a(k)]`).

    Note
    -------
    :math:`\\hat{Q}_k := \\hat{Q}_k(x, a(1), \\ldots, a(k))` is recursively derived as follows.

    .. :math:
        \\hat{Q}_k \\leftarrow \\argmin_{Q_k} \\mathbb{E}_T [ w_{1:k} Q_k(x, a(1), \\ldots, a(k))
            - ( \\alpha(k) r(k) + \\mathbb{E}_{a'(k+1)} \\hat{Q}_{k+1}(x, a(1), \\ldots, a(k), a'(k+1)) ) ]

    where :math:`\\mathcal{D}=\\{(x_t,a_t,r_t)\\}_{t=1}^{T}` is logged bandit feedback data with :math:`T` rounds collected by
    a behavior policy :math:`\\pi_b`. Both :math:`a_t` and :math:`r_t` vectors that :math:`a_t(k)` and :math:`r_t(k)` denote the action and the reward
    presented at slot :math:`k` (where a slate consists of :math:`K` slots (slate size)). :math:`\\alpha(k)` is a non-negative weight at slot :math:`k`.
    We denote :math:`w_{1:k} := \\prod_{k'=1}^k \\pi_e(a(k') | x, a(1), \\ldots, a(k'-1)) / \\pi_b(a(k') | x, a(1), \\ldots, a(k'-1))` and
    :math:`\\hat{Q}_k := \\hat{Q}_k(x, a(1), \\ldots, a(k))`.
    Finally, :math:`\\mathbb{E}_T [ \\cdot ]` is empirical average over :math:`\\mathcal{D}` and
    :math:`\\mathbb{E}_{a'(k)} [ \\cdot ] := \\mathbb{E}_{a'(k) \\sim \\pi_e(a'(k) | x, a(1), \\ldots, a(k-1))} [ \\cdot ]`.

    Parameters
    ------------
    base_model: BaseEstimator
        A machine learning model used to estimate the mean reward function.

    len_list: int
        Length of a list of actions recommended in each impression (slate size).
        When Open Bandit Dataset is used, 3 should be set.

    n_unique_action: int
        Number of unique actions.

    fitting_method: str, default='normal'
        Method to fit the regression model.
        Must be either of ['normal', 'iw'] where 'iw' stands for importance weighting.

    Reference
    ------------
    Haruka Kiyohara, Yuta Saito, Tatsuya Matsuhiro, Yusuke Narita, Nobuyuki Shimizu, and Yasuo Yamamoto.
    "Doubly Robust Off-Policy Evaluation for Ranking Policies under the Cascade Behavior Model.", 2021.

    """

    base_model: BaseEstimator
    len_list: int
    n_unique_action: int
    fitting_method: str = "normal"

    def __post_init__(self):
        """Initialize Class."""
        self.base_model_list = [clone(self.base_model) for _ in range(self.len_list)]
        self.action_context = np.eye(self.n_unique_action)

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
    ):
        """Fit the regression model on given logged bandit feedback data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like, (n_rounds * len_list,)
            Action observed at each slot in each round of the logged bandit feedback, i.e., :math:`a_{t}(k)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        pscore_cascade: array-like, shape (n_rounds * len_list,)
            Probabilities that behavior policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_b(a_t(k) | x_t, a_t(1), \\ldots, a_t(k-1))`.

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Probabilities that evaluation policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_e(a_t(k) | x_t, a_t(1), \\ldots, a_t(k-1))`.

        evaluation_policy_action_dist: array-like (n_rounds * len_list * n_unique_actions, )
            Action choice probabilities of evaluation policy for all possible actions
            , i.e., :math:`\\pi_e({a'}_t(k) | x_t, a_t(1), \\ldots, a_t(k-1)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

        """
        # (n_rounds_ * len_list, ) -> (n_rounds_, len_list)
        action = action.reshape((-1, self.len_list))
        reward = reward.reshape((-1, self.len_list))
        iw = (evaluation_policy_pscore_cascade / pscore_cascade).reshape(
            (-1, self.len_list)
        )

        # (n_rounds_, )
        n_rounds_ = len(action)
        sample_weight = np.ones(n_rounds_)

        for position_ in range(self.len_list)[::-1]:
            X, y = self._preprocess_for_reg_model(
                context=context,
                action=action,
                reward=reward,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
                position_=position_,
            )

            if self.fitting_method == "iw":
                sample_weight = iw[:, position_]

            self.base_model_list[position_].fit(X, y, sample_weight=sample_weight)

    def predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
    ):
        """Predict the baseline values.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        action: array-like, shape (n_rounds_of_new_data * len_list, )
            Action vectors for new data.

        Returns
        -----------
        q_hat_for_counterfactual_actions: array-like, shape (n_rounds_of_new_data * len_list * n_unique_action, )
            Estimated baseline values for new data by the regression model.

        """
        n_rounds_of_new_data = len(context)
        # (n_rounds_of_new_data * len_list, ) -> (n_rounds_of_new_data, len_list)
        action = action.reshape((-1, self.len_list))
        # (n_rounds_, len_list, n_unique_action, )
        q_hat_for_counterfactual_actions = np.zeros(
            (n_rounds_of_new_data, self.len_list, self.n_unique_action)
        )
        for position_ in range(self.len_list)[::-1]:
            # the action vector shrinks every time as the position_ decreases
            # (n_rounds_of_new_data, position_ - 1)
            action = action[:, :position_]
            # (n_rounds_of_new_data, dim_context) -> (n_rounds_of_new_data * n_unique_action, dim_context)
            context_ = []
            # (n_rounds_of_new_data, position_) -> (n_rounds_of_new_data * n_unique_action, position_)
            action_ = []
            for i in range(n_rounds_of_new_data):
                for a_ in range(self.n_unique_action):
                    context_.append(context[i])
                    action_.append(np.append(action[i], a_))
            # (n_rounds_of_new_data * n_unique_action, dim_context + position_)
            X = np.concatenate([context_, action_], axis=1)
            # (n_rounds_of_new_data * n_unique_action, ) -> (n_rounds_of_new_data, n_unique_action)
            q_hat_for_counterfactual_actions[:, position_, :] = (
                self.base_model_list[position_]
                .predict(X)
                .reshape((-1, self.n_unique_action))
            )
        # (n_rounds_of_new_data * len_list * n_unique_action, )
        return q_hat_for_counterfactual_actions.flatten()

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
    ):
        """Fit the regression model on given logged bandit feedback data and predict the reward function of the same data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like, (n_rounds * len_list,)
            Action observed at each slot in each round of the logged bandit feedback, i.e., :math:`a_{t}(k)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds * len_list,)
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        pscore_cascade: array-like, shape (n_rounds * len_list,)
            Probabilities that behavior policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_b(a_t(k) | x_t, a_t(1), \\ldots, a_t(k-1))`.

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Probabilities that evaluation policy selects action :math:`a` at position (slot) `k` conditional on the previous actions (presented at position `1` to `k-1`)
            , i.e., :math:`\\pi_e(a_t(k) | x_t, a_t(1), \\ldots, a_t(k-1))`.

        evaluation_policy_action_dist: array-like (n_rounds * len_list * n_unique_actions, )
            Action choice probabilities of evaluation policy for all possible actions
            , i.e., :math:`\\pi_e({a'}_t(k) | x_t, a_t(1), \\ldots, a_t(k-1)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

        Returns
        -----------
        q_hat_for_counterfactual_actions: array-like, shape (n_rounds_of_new_data * len_list * n_unique_action, )
            Estimated baseline values for new data by the regression model.

        """
        self.fit(
            context=context,
            action=action,
            reward=reward,
            pscore_cascade=pscore_cascade,
            evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
            evaluation_policy_action_dist=evaluation_policy_action_dist,
        )
        # (n_rounds_test, len_list, n_unique_action, )
        return self.predict(context=context, action=action)

    def _preprocess_for_reg_model(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
        position_: int,
    ):
        """Preprocess feature vectors and estimation target to train a give regression model.

        Note
        -----
        Please override this method if you want to use another feature enginnering
        for training the regression model.

        Parameters
        -----------
        context: array-like, shape (n_rounds_, dim_context)
            Context vectors in the training logged bandit feedback.

        action: array-like, (n_rounds_ * len_list, )
            Action observed at each slot in each round of the logged bandit feedback, i.e., :math:`a_{t}(k)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds_ * len_list, )
            Reward observed at each slot in each round of the logged bandit feedback, i.e., :math:`r_{t}(k)`.

        evaluation_policy_action_dist: array-like (n_rounds_ * len_list * n_unique_actions, )
            Action choice probabilities of evaluation policy for all possible actions
            , i.e., :math:`\\pi_e({a'}_t(k) | x_t, a_t(1), \\ldots, a_t(k-1)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

        position_: int
            Position id (slot) in a slate.

        Returns
        -----------
        X, y: array-like, shape(n_rounds, )
            Input and target vectors in prediction.

        """
        n_rounds_ = len(context)
        # (n_rounds_, len_list) -> (n_rounds_, position_)
        action = action[:, : position_ + 1]
        # (n_rounds_, len_list) -> (n_rounds_, )
        reward = reward[:, position_]
        # estimator input
        X = np.concatenate([context, action], axis=1)
        # estimate baseline value at the next position
        # (n_rounds_, )
        if position_ + 1 == self.len_list:
            q_hat_at_next_position = np.zeros(n_rounds_)
        else:
            # (n_rounds_ * len_list * n_unique_action, ) -> (n_rounds_, len_list, n_unique_action) -> (n_rounds_, len_list) -> (n_rounds_ * n_unique_action, )
            evaluation_policy_action_dist_at_next_position = (
                evaluation_policy_action_dist.reshape(
                    (-1, self.len_list, self.n_unique_action)
                )[:, position_ + 1, :]
            ).flatten()
            # (n_rounds_, dim_context) -> (n_rounds_ * n_unique_action, dim_context)
            context_ = []
            # (n_rounds_, position_ + 1) -> (n_rounds_ * n_unique_action, position_ + 1)
            action_ = []
            for i in range(n_rounds_):
                for a_ in range(self.n_unique_action):
                    context_.append(context[i])
                    action_.append(np.append(action[i], a_))
            X_ = np.concatenate([context_, action_], axis=1)
            # (n_rounds_ * n_unique_action, ) -> (n_rounds_, )
            q_hat_for_counterfactual_actions_at_next_position = self.base_model_list[
                position_ + 1
            ].predict(X_)
            # baseline estimation by evaluation policy
            # (n_rounds_ * n_unique_action, ) -> (n_rounds_, n_unique_action) -> (n_rounds_, )
            q_hat_at_next_position = (
                (
                    evaluation_policy_action_dist_at_next_position
                    * q_hat_for_counterfactual_actions_at_next_position
                )
                .reshape((-1, self.n_unique_action))
                .sum(axis=1)
            )
        # (n_rounds_, )
        y = reward + q_hat_at_next_position
        # (n_rounds_, dim_context + position_), (n_rounds_, )
        return X, y
