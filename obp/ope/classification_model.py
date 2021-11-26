# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Regression Model Class for Estimating Mean Reward Functions."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
from sklearn.calibration import CalibratedClassifierCV

from ..utils import check_bandit_feedback_inputs, sample_action_fast


@dataclass
class ImportanceSampler(BaseEstimator):
    """Machine learning model to distinguish between the behavior and evaluation policy (:math:`\\Pr[C=1 | x, a]`),
    where :math:`C` equals to 1 if the action is sampled by evaluation policy.

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

    fitting_method: str, default='weighted_loss'
        Method to fit the regression model.
        Must be one of ['weighted_loss', 'sample', 'raw']. Each method is defined as follows:
            - weighted_loss: for each round, n_actions rows are duplicated. For each duplicated row, action features are represented by one-hot encoding of each action. Classification models are trained with sample_weight, where sample_weight is the probability that the corresponding action is sampled (action_dist_at_position[:, action_idx]).
            - sample: actions are sampled by applying Gumbel-softmax trick to action_dist_at_position, and action features are represented by one-hot encoding of the sampled action.
            - raw: action_dist_at_position are directly encoded as action features.

    References
    -----------
    Arjun Sondhi, David Arbour, and Drew Dimmery
    "Balanced Off-Policy Evaluation in General Action Spaces.", 2020.

    """

    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "weighted_loss"
    fitting_random_state: Optional[int] = None
    calibration_cv: int = 2

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["weighted_loss", "sample", "raw"]
        ):
            raise ValueError(
                f"fitting_method must be one of 'weighted_loss', 'sample', or 'raw', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "base_model must be BaseEstimator or a child class of BaseEstimator"
            )

        if self.calibration_cv > 0:
            self.base_model_list = [
                clone(
                    CalibratedClassifierCV(
                        base_estimator=self.base_model, cv=self.calibration_cv
                    ),
                )
                for _ in np.arange(self.len_list)
            ]
        else:
            self.base_model_list = [
                clone(self.base_model) for _ in np.arange(self.len_list)
            ]
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
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
            If None is given, a regression model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.

        action_dist: array-like, shape (n_rounds, n_actions, len_list), default=None
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.
            When either of 'iw' or 'mrdr' is used as the 'fitting_method' argument, then `action_dist` must be given.

        """
        # check_bandit_feedback_inputs(
        #     context=context,
        #     action=action,
        #     reward=reward,
        #     pscore=pscore,
        #     position=position,
        #     action_context=self.action_context,
        # )
        n_rounds = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"position elements must be smaller than len_list, but the maximum value is {position.max()} (>= {self.len_list})"
                )
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

        # If self.fitting_method != "sample", `sampled_action` has no information
        sampled_action = np.zeros((n_rounds, self.n_actions, self.len_list))
        if self.fitting_method == "sample":
            for position_ in np.arange(self.len_list):
                idx = position == position_
                sampled_action_at_position = sample_action_fast(
                    action_dist=action_dist[idx][:, :, position_],
                    random_state=self.fitting_random_state,
                )
                sampled_action[
                    idx,
                    sampled_action_at_position,
                    position_,
                ] = 1

        for position_ in np.arange(self.len_list):
            idx = position == position_
            action_dist_at_position = action_dist[idx][:, :, position_]
            X, y, sample_weight = self._pre_process_for_clf_model(
                context=context[idx],
                action=action[idx],
                action_dist_at_position=action_dist_at_position,
                sampled_action_at_position=sampled_action[idx][:, :, position_],
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {position_}")
            self.base_model_list[position_].fit(X, y, sample_weight=sample_weight)

    def predict(
        self,
        action: np.ndarray,
        context: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        proba_of_evaluation_policy = np.zeros(action.shape[0])
        for position_ in np.arange(self.len_list):
            idx = position == position_
            X, _, _ = self._pre_process_for_clf_model(
                context=context[idx],
                action=action[idx],
                is_prediction=True,
            )
            proba_of_evaluation_policy[idx] = self.base_model_list[
                position_
            ].predict_proba(X)[:, 1]
        return proba_of_evaluation_policy / (1 - proba_of_evaluation_policy)

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        n_folds: int = 1,
        random_state: Optional[int] = None,
        is_eval_model: bool = False,
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
            If None is given, a regression model assumes that there is only one position.
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
        # check_bandit_feedback_inputs(
        #     context=context,
        #     action=action,
        #     reward=reward,
        #     pscore=pscore,
        #     position=position,
        #     action_context=self.action_context,
        # )
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

        if n_folds == 1:
            self.fit(
                context=context,
                action=action,
                position=position,
                action_dist=action_dist,
            )
            return self.predict(context=context, action=action, position=position)
        else:
            balancing_weight = np.zeros(n_rounds)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        if is_eval_model:
            self.eval_result = {"y": [], "proba": [], "sample_weight": []}
        for train_idx, test_idx in kf.split(context):
            self.fit(
                context=context[train_idx],
                action=action[train_idx],
                position=position[train_idx],
                action_dist=action_dist[train_idx],
            )
            balancing_weight[test_idx] = self.predict(
                context=context[test_idx],
                action=action[test_idx],
                position=position[test_idx],
            )
            if is_eval_model:
                sampled_action = np.zeros(
                    (test_idx.shape[0], self.n_actions, self.len_list)
                )
                if self.fitting_method == "sample":
                    for position_ in np.arange(self.len_list):
                        idx = position[test_idx] == position_
                        sampled_action_at_position = sample_action_fast(
                            action_dist=action_dist[test_idx][idx][:, :, position_],
                            random_state=self.fitting_random_state,
                        )
                        sampled_action[
                            idx,
                            sampled_action_at_position,
                            position_,
                        ] = 1
                for position_ in np.arange(self.len_list):
                    idx = position[test_idx] == position_
                    action_dist_at_position = action_dist[test_idx][idx][
                        :, :, position_
                    ]
                    X, y, sample_weight = self._pre_process_for_clf_model(
                        context=context[test_idx][idx],
                        action=action[test_idx][idx],
                        action_dist_at_position=action_dist_at_position,
                        sampled_action_at_position=sampled_action[idx][:, :, position_],
                    )
                    proba_of_evaluation_policy = self.base_model_list[
                        position_
                    ].predict_proba(X)[:, 1]
                    self.eval_result["proba"].append(proba_of_evaluation_policy)
                    self.eval_result["y"].append(y)
                    self.eval_result["sample_weight"].append(sample_weight)
        return balancing_weight

    def _pre_process_for_clf_model(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_dist_at_position: Optional[np.ndarray] = None,
        sampled_action_at_position: Optional[np.ndarray] = None,
        is_prediction: bool = False,
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
        behavior_feature = np.c_[context, self.action_context[action]]
        if is_prediction:
            return behavior_feature, None, None
        if self.fitting_method == "weighted_loss":
            X = np.copy(behavior_feature)
            y = np.zeros(X.shape[0], dtype=int)
            sample_weight = np.ones(X.shape[0])
            for action_idx in np.arange(self.n_actions):
                tmp_action = np.ones(context.shape[0], dtype=int) * action_idx
                evaluation_feature = np.c_[context, self.action_context[tmp_action]]
                X = np.r_[X, evaluation_feature]
                y = np.r_[y, np.ones(evaluation_feature.shape[0], dtype=int)]
                sample_weight = np.r_[
                    sample_weight, action_dist_at_position[:, action_idx]
                ]
        else:
            if self.fitting_method == "raw":
                evaluation_feature = np.c_[context, action_dist_at_position]
            elif self.fitting_method == "sample":
                evaluation_feature = np.c_[context, sampled_action_at_position]
            X = np.copy(behavior_feature)
            y = np.zeros(X.shape[0], dtype=int)
            X = np.r_[X, evaluation_feature]
            y = np.r_[y, np.ones(evaluation_feature.shape[0], dtype=int)]
            sample_weight = None
        return X, y, sample_weight
