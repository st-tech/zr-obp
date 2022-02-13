# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Regression Model Class for Estimating the Q functions in Cascade-DR."""
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.utils import check_scalar

from obp.utils import check_array


@dataclass
class SlateRegressionModel(BaseEstimator):
    """Machine learning model to estimate the Q functions for OPE of ranking policies.

    Note
    -------
    Q function at position :math:`l` is defined as
    :math:`\\hat{Q}_l := \\hat{Q}_l(x, a(1), \\ldots, a(k)) \\approx \\mathbb{E}[ \\sum_{l'=l}^L \\alpha_{l'} r(l') | x, a(1), \\ldots, a(l)]`).

    Q function is estimated recursively, and then used in Cascade-DR.
    Please refer to Section 3.1 of Kiyohara et al.(2022) for more details.

    Parameters
    ------------
    base_model: BaseEstimator
        A machine learning model used to estimate the Q function.

    len_list: int
        Length of a list of actions in a recommendation/ranking interface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    n_unique_action: int
        Number of unique actions.

    fitting_method: str, default='normal'
        Method to fit the regression model.
        Must be either of ['normal', 'iw'] where 'iw' stands for importance weighting.

    Reference
    ------------
    Haruka Kiyohara, Yuta Saito, Tatsuya Matsuhiro, Yusuke Narita, Nobuyuki Shimizu, and Yasuo Yamamoto.
    "Doubly Robust Off-Policy Evaluation for Ranking Policies under the Cascade Behavior Model.", 2022.

    """

    base_model: BaseEstimator
    len_list: int
    n_unique_action: int
    fitting_method: str = "normal"

    def __post_init__(self):
        """Initialize Class."""
        check_scalar(self.n_unique_action, "n_unique_action", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["normal", "iw"]
        ):
            raise ValueError(
                f"`fitting_method` must be either 'normal' or 'iw', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "`base_model` must be BaseEstimator or a child class of BaseEstimator"
            )
        if is_classifier(self.base_model):
            raise ValueError("`base_model` must be a regressor, not a classifier")
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
        """Fit the regression model on given logged bandit data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, (n_rounds * len_list,)
            Actions observed at each slot in a ranking/slate in logged bandit data, i.e., :math:`a_{i}(l)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds * len_list,)
            Slot-level rewards observed for each data in logged bandit data, i.e., :math:`r_{i}(l)`.

        pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`).

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_action_dist: array-like (n_rounds * len_list * n_unique_actions, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e({a'}_t(k) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall {a'}_t(k) \\in \\mathcal{A}`.

        """
        check_array(array=context, name="context", expected_dim=2)
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=pscore_cascade, name="pscore_cascade", expected_dim=1)
        check_array(
            array=evaluation_policy_pscore_cascade,
            name="evaluation_policy_pscore_cascade",
            expected_dim=1,
        )
        check_array(
            array=evaluation_policy_action_dist,
            name="evaluation_policy_action_dist",
            expected_dim=1,
        )
        if not (
            action.shape
            == reward.shape
            == pscore_cascade.shape
            == evaluation_policy_pscore_cascade.shape
            == (context.shape[0] * self.len_list,)
        ):
            raise ValueError(
                "Expected `action.shape == reward.shape == pscore_cascade.shape == evaluation_policy_pscore_cascade.shape"
                " == (context.shape[0] * len_list, )`"
                ", but found it False"
            )
        if evaluation_policy_action_dist.shape != (
            context.shape[0] * self.len_list * self.n_unique_action,
        ):
            raise ValueError(
                "Expected `evaluation_policy_action_dist.shape == (context.shape[0] * len_list * n_unique_action, )`"
                ", but found it False"
            )
        if not (
            np.issubdtype(action.dtype, np.integer)
            and action.min() >= 0
            and action.max() < self.n_unique_action
        ):
            raise ValueError(
                "`action` elements must be integers in the range of [0, n_unique_action)"
            )
        if np.any(pscore_cascade <= 0) or np.any(pscore_cascade > 1):
            raise ValueError("`pscore_cascade` must be in the range of (0, 1]")
        if np.any(evaluation_policy_pscore_cascade <= 0) or np.any(
            evaluation_policy_pscore_cascade > 1
        ):
            raise ValueError(
                "`evaluation_policy_pscore_cascade` must be in the range of (0, 1]"
            )
        if not np.allclose(
            np.ones(
                evaluation_policy_action_dist.reshape((-1, self.n_unique_action)).shape[
                    0
                ]
            ),
            evaluation_policy_action_dist.reshape((-1, self.n_unique_action)).sum(
                axis=1
            ),
        ):
            raise ValueError(
                "`evaluation_policy_action_dist[i * n_unique_action : (i+1) * n_unique_action]` "
                "must sum up to one for all i."
            )
        # (n_rounds_ * len_list, ) -> (n_rounds_, len_list)
        action = action.reshape((-1, self.len_list))
        reward = reward.reshape((-1, self.len_list))
        iw = (evaluation_policy_pscore_cascade / pscore_cascade).reshape(
            (-1, self.len_list)
        )

        # (n_rounds_, )
        n_rounds_ = len(action)
        sample_weight = np.ones(n_rounds_)

        for pos_ in range(self.len_list)[::-1]:
            X, y = self._preprocess_for_reg_model(
                context=context,
                action=action,
                reward=reward,
                evaluation_policy_action_dist=evaluation_policy_action_dist,
                position_=pos_,
            )

            if self.fitting_method == "iw":
                sample_weight = iw[:, pos_]

            self.base_model_list[pos_].fit(X, y, sample_weight=sample_weight)

    def predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
    ):
        """Predict the Q function values.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        action: array-like, shape (n_rounds_of_new_data * len_list, )
            Action vectors for new data.

        Returns
        -----------
        q_hat: array-like, shape (n_rounds_of_new_data * len_list * n_unique_action, )
            Estimated Q function values of new data.
            :math:`\\hat{Q}_{i,l}(x_i, a_i(1), \\ldots, a_i(l-1), a_i(l)) \\forall a_i(l) \\in \\mathcal{A}`.

        """
        check_array(array=context, name="context", expected_dim=2)
        check_array(array=action, name="action", expected_dim=1)
        if action.shape != (context.shape[0] * self.len_list,):
            raise ValueError(
                "Expected `action.shape == (context.shape[0] * len_list, )`"
                ", but found it False"
            )
        n_rounds_of_new_data = len(context)
        # (n_rounds_of_new_data * len_list, ) -> (n_rounds_of_new_data, len_list)
        action = action.reshape((-1, self.len_list))
        # (n_rounds_, len_list, n_unique_action, )
        q_hat = np.zeros((n_rounds_of_new_data, self.len_list, self.n_unique_action))
        for pos_ in range(self.len_list)[::-1]:
            # the action vector shrinks every time as the position_ decreases
            # (n_rounds_of_new_data, position_ - 1)
            action = action[:, :pos_]
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
            q_hat[:, pos_, :] = (
                self.base_model_list[pos_]
                .predict(X)
                .reshape((-1, self.n_unique_action))
            )
        # (n_rounds_of_new_data * len_list * n_unique_action, )
        return q_hat.flatten()

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore_cascade: np.ndarray,
        evaluation_policy_pscore_cascade: np.ndarray,
        evaluation_policy_action_dist: np.ndarray,
    ):
        """Fit the regression model on given logged bandit data and predict the Q function values on the same data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, (n_rounds * len_list,)
            Actions observed at each slot in a ranking/slate in logged bandit data, i.e., :math:`a_{i}(l)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds * len_list,)
            Slot-level rewards observed for each data in logged bandit data, i.e., :math:`r_{i}(l)`.

        pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`).

        evaluation_policy_pscore_cascade: array-like, shape (n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a particular sequence of actions from the top position to the :math:`l`-th position (:math:`a_{1:l}`). This type of action choice probabilities corresponds to the cascade model.

        evaluation_policy_action_dist: array-like (n_rounds * len_list * n_unique_actions, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

        Returns
        -----------
        q_hat: array-like, shape (n_rounds_of_new_data * len_list * n_unique_action, )
            Estimated Q functions for new data by the regression model.

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
        """Preprocess feature vectors and target variables for training a regression model.

        Note
        -----
        Please override this method if you want to use another feature enginnering
        for training the regression model.

        Parameters
        -----------
        context: array-like, shape (n_rounds_, dim_context)
            Context vectors in the training set of logged bandit data.

        action: array-like, (n_rounds_ * len_list, )
            Actions observed at each slot in a ranking/slate in logged bandit data, i.e., :math:`a_{i}(l)`,
            which is chosen by the behavior policy :math:`\\pi_b`.

        reward: array-like, shape (n_rounds_ * len_list, )
            Slot-level rewards observed for each data in logged bandit data, i.e., :math:`r_{i}(l)`.

        evaluation_policy_action_dist: array-like (n_rounds_ * len_list * n_unique_actions, )
            Plackett-luce style action distribution induced by evaluation policy
            (action choice probabilities at each slot given previous action choices)
            , i.e., :math:`\\pi_e(a_i(l) | x_i, a_i(1), \\ldots, a_i(l-1)) \\forall a_i(l) \\in \\mathcal{A}`.

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
        # estimate the Q function at the next position
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
            q_hat_at_next_position = self.base_model_list[position_ + 1].predict(X_)
            # the expected Q function under the evaluation policy
            # (n_rounds_ * n_unique_action, ) -> (n_rounds_, n_unique_action) -> (n_rounds_, )
            q_hat_at_next_position = (
                (
                    evaluation_policy_action_dist_at_next_position
                    * q_hat_at_next_position
                )
                .reshape((-1, self.n_unique_action))
                .sum(axis=1)
            )
        # (n_rounds_, )
        y = reward + q_hat_at_next_position
        # (n_rounds_, dim_context + position_), (n_rounds_, )
        return X, y
