# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Classification Model Class for Estimating Propensity Score and Importance Weight."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from ..utils import check_array
from ..utils import check_bandit_feedback_inputs
from ..utils import sample_action_fast


@dataclass
class ImportanceWeightEstimator(BaseEstimator):
    """Machine learning model to estimate the importance weights induced by the behavior and evaluation policies leveraging  classifier-based density ratio estimation.

    Parameters
    ------------
    base_model: BaseEstimator
        A machine learning model used to estimate the importance weights.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    action_context: array-like, shape (n_actions, dim_action_context), default=None
        Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        If None, one-hot encoding of the action variable is used as default.
        If fitting_method is 'raw', one-hot encoding will be used as action_context.

    fitting_method: str, default='sample'
        Method to fit the classification model.
        Must be one of ['sample', 'raw']. Each method is defined as follows:
            - sample: actions are sampled from behavior and evaluation policies, respectively.
            - raw: action_dist_at_pos are directly encoded as action features.
        If fitting_method is 'raw', one-hot encoding will be used as action_context.

    calibration_cv: int, default=2
        Number of folds in the calibration procedure.
        If calibration_cv <= 1, classification model is not calibrated.

    References
    -----------
    Arjun Sondhi, David Arbour, and Drew Dimmery
    "Balanced Off-Policy Evaluation in General Action Spaces.", 2020.

    """

    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "sample"
    calibration_cv: int = 2

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        check_scalar(self.calibration_cv, "calibration_cv", int)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["sample", "raw"]
        ):
            raise ValueError(
                f"`fitting_method` must be either 'sample' or 'raw', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "`base_model` must be BaseEstimator or a child class of BaseEstimator"
            )

        if self.calibration_cv > 1:
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
        if self.action_context is None or self.fitting_method == "raw":
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit the classification model on given logged bandit data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a classification model assumes that there is only a single position in  a recommendation interface.
            When `len_list` > 1, an array must be given as `position`.

        random_state: int, default=None
            `random_state` affects the sampling of actions from the evaluation policy.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=np.zeros_like(action),  # use dummy reward
            position=position,
            action_context=self.action_context,
        )
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        n = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            check_array(array=position, name="position", expected_dim=1)
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if action_dist.shape != (n, self.n_actions, self.len_list):
            raise ValueError(
                f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n, self.n_actions, self.len_list}), but is {action_dist.shape}"
            )
        if not np.allclose(action_dist.sum(axis=1), 1):
            raise ValueError("`action_dist` must be a probability distribution")

        # If self.fitting_method != "sample", `sampled_action` has no information
        sampled_action = np.zeros(n, dtype=int)
        if self.fitting_method == "sample":
            for pos_ in np.arange(self.len_list):
                idx = position == pos_
                sampled_action_at_position = sample_action_fast(
                    action_dist=action_dist[idx][:, :, pos_],
                    random_state=random_state,
                )
                sampled_action[idx] = sampled_action_at_position

        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            action_dist_at_pos = action_dist[idx][:, :, pos_]
            X, y = self._pre_process_for_clf_model(
                context=context[idx],
                action=action[idx],
                action_dist_at_pos=action_dist_at_pos,
                sampled_action_at_position=sampled_action[idx],
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {pos_}")
            self.base_model_list[pos_].fit(X, y)

    def predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict the importance weights.

        Parameters
        ----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds_of_new_data,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        position: array-like, shape (n_rounds_of_new_data,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a classification model assumes that there is only a single position in  a recommendation interface.
            When `len_list` > 1, an array must be given as `position`.

        Returns
        ----------
        estimated_importance_weights: array-like, shape (n_rounds_of_new_data, )
            Importance weights estimated via supervised classification, i.e., :math:`\\hat{w}(x_t, a_t)`.

        """
        proba_eval_policy = np.zeros(action.shape[0])
        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            X, _, = self._pre_process_for_clf_model(
                context=context[idx],
                action=action[idx],
                is_prediction=True,
            )
            proba_eval_policy[idx] = self.base_model_list[pos_].predict_proba(X)[:, 1]
        return proba_eval_policy / (1 - proba_eval_policy)

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        n_folds: int = 1,
        random_state: Optional[int] = None,
        evaluate_model_performance: bool = False,
    ) -> np.ndarray:
        """Fit the classification model on given logged bandit data and predict the importance weights on the same data, possibly using cross-fitting to avoid over-fitting.

        Note
        ------
        When `n_folds` is larger than 1, the cross-fitting procedure is applied.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a classification model assumes that there is only a single position in  a recommendation interface.
            When `len_list` > 1, an array must be given as `position`.

        n_folds: int, default=1
            Number of folds in the cross-fitting procedure.
            When 1 is given, the classification model is trained on the whole logged bandit data.
            Please refer to https://arxiv.org/abs/2002.08536 about the details of the cross-fitting procedure.

        random_state: int, default=None
            `random_state` affects the ordering of the indices, which controls the randomness of each fold.
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html for the details.

        evaluate_model_performance: bool, default=False
            Whether the performance of the classification model is evaluated or not.
            If True, the predicted probability of the classification model and the true label of each fold is saved in `self.eval_result[fold]`

        Returns
        -----------
        estimated_importance_weights: array-like, shape (n_rounds_of_new_data, )
            Importance weights estimated via supervised classification, i.e., :math:`\\hat{w}(x_t, a_t)`.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=np.zeros_like(action),  # use dummy reward
            position=position,
            action_context=self.action_context,
        )
        n = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )

        check_array(array=action_dist, name="action_dist", expected_dim=3)
        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if action_dist.shape != (n, self.n_actions, self.len_list):
            raise ValueError(
                f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n, self.n_actions, self.len_list}), but is {action_dist.shape}"
            )
        if not np.allclose(action_dist.sum(axis=1), 1):
            raise ValueError("`action_dist` must be a probability distribution")

        if n_folds == 1:
            self.fit(
                context=context,
                action=action,
                position=position,
                action_dist=action_dist,
                random_state=random_state,
            )
            return self.predict(context=context, action=action, position=position)
        else:
            estimated_importance_weights = np.zeros(n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        if evaluate_model_performance:
            self.eval_result = {"y": [], "proba": []}
        for train_idx, test_idx in kf.split(context):
            self.fit(
                context=context[train_idx],
                action=action[train_idx],
                position=position[train_idx],
                action_dist=action_dist[train_idx],
                random_state=random_state,
            )
            estimated_importance_weights[test_idx] = self.predict(
                context=context[test_idx],
                action=action[test_idx],
                position=position[test_idx],
            )
            if evaluate_model_performance:
                sampled_action = np.zeros(test_idx.shape[0], dtype=int)
                if self.fitting_method == "sample":
                    for pos_ in np.arange(self.len_list):
                        idx = position[test_idx] == pos_
                        sampled_action_at_position = sample_action_fast(
                            action_dist=action_dist[test_idx][idx][:, :, pos_],
                            random_state=random_state,
                        )
                        sampled_action[idx] = sampled_action_at_position
                for pos_ in np.arange(self.len_list):
                    idx = position[test_idx] == pos_
                    action_dist_at_pos = action_dist[test_idx][idx][:, :, pos_]
                    X, y = self._pre_process_for_clf_model(
                        context=context[test_idx][idx],
                        action=action[test_idx][idx],
                        action_dist_at_pos=action_dist_at_pos,
                        sampled_action_at_position=sampled_action[idx],
                    )
                    proba_eval_policy = self.base_model_list[pos_].predict_proba(X)[
                        :, 1
                    ]
                    self.eval_result["proba"].append(proba_eval_policy)
                    self.eval_result["y"].append(y)
        return estimated_importance_weights

    def _pre_process_for_clf_model(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_dist_at_pos: Optional[np.ndarray] = None,
        sampled_action_at_position: Optional[np.ndarray] = None,
        is_prediction: bool = False,
    ) -> np.ndarray:
        """Preprocess feature vectors and output labels to train a classification model.

        Note
        -----
        Please override this method if you want to use another feature enginnering
        for training the classification model.

        Parameters
        -----------
        context: array-like, shape (n_rounds,)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.


        action_dist_at_pos: array-like, shape (n_rounds, n_actions,)
            Action choice probabilities of the evaluation policy of each position (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        sampled_action_at_position: array-like, shape (n_rounds, n_actions,)
            Actions sampled by evaluation policy for each data at each position.

        """
        behavior_policy_feature = np.c_[context, self.action_context[action]]
        if is_prediction:
            return behavior_policy_feature, None
        if self.fitting_method == "raw":
            evaluation_policy_feature = np.c_[context, action_dist_at_pos]
        elif self.fitting_method == "sample":
            evaluation_policy_feature = np.c_[
                context, self.action_context[sampled_action_at_position]
            ]
        X = np.copy(behavior_policy_feature)
        y = np.zeros(X.shape[0], dtype=int)
        X = np.r_[X, evaluation_policy_feature]
        y = np.r_[y, np.ones(evaluation_policy_feature.shape[0], dtype=int)]
        return X, y


@dataclass
class PropensityScoreEstimator(BaseEstimator):
    """Machine learning model to estimate propensity scores (:math:`\\pi_{b}(a|x)`).

    Parameters
    ------------
    base_model: BaseEstimator
        A machine learning model used to estimate the reward function.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    calibration_cv: int, default=2
        Number of folds in the calibration procedure.
        If calibration_cv <= 1, calibration will not be applied.

    References
    -----------
    Arjun Sondhi, David Arbour, and Drew Dimmery
    "Balanced Off-Policy Evaluation in General Action Spaces.", 2020.

    """

    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    calibration_cv: int = 2

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        check_scalar(self.calibration_cv, "calibration_cv", int)
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "`base_model` must be BaseEstimator or a child class of BaseEstimator"
            )

        if self.calibration_cv > 1:
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

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the classification model on given logged bandit data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a classification model assumes that there is only a single position in  a recommendation interface.
            When `len_list` > 1, an array must be given as `position`.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=np.zeros_like(action),  # use dummy reward
            position=position,
            action_context=np.eye(self.n_actions, dtype=int),
        )

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )

        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            if context[idx].shape[0] == 0:
                raise ValueError(f"No training data at position {pos_}")
            self.base_model_list[pos_].fit(context[idx], action[idx])

    def predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict the propensity scores.

        Parameters
        ----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds_of_new_data,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        position: array-like, shape (n_rounds_of_new_data,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a classification model assumes that there is only a single position in  a recommendation interface.
            When `len_list` > 1, an array must be given as `position`.

        Returns
        ----------
        estimated_pscore: array-like, shape (n_rounds_of_new_data, )
            Estimated propensity scores, i.e., :math:`\\hat{\\pi}_b (a \\mid x)`.

        """
        estimated_pscore = np.zeros(action.shape[0])
        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            if context[idx].shape[0] == 0:
                continue
            estimated_pscore[idx] = self.base_model_list[pos_].predict_proba(
                context[idx]
            )[np.arange(action[idx].shape[0]), action[idx]]
        return estimated_pscore

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        position: Optional[np.ndarray] = None,
        n_folds: int = 1,
        random_state: Optional[int] = None,
        evaluate_model_performance: bool = False,
    ) -> np.ndarray:
        """Fit the classification model on given logged bandit data and predict the propensity score on the same data, possibly using the cross-fitting procedure to avoid over-fitting.

        Note
        ------
        When `n_folds` is larger than 1, the cross-fitting procedure is applied.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a classification model assumes that there is only a single position.
            When `len_list` > 1, an array must be given as `position`.

        n_folds: int, default=1
            Number of folds in the cross-fitting procedure.
            When 1 is given, the classification model is trained on the whole logged bandit data.
            Please refer to https://arxiv.org/abs/2002.08536 about the details of the cross-fitting procedure.

        random_state: int, default=None
            `random_state` affects the ordering of the indices, which controls the randomness of each fold.
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html for the details.

        evaluate_model_performance: bool, default=False
            Whether the performance of the classification model is evaluated or not.
            If True, the predicted probability of the classification model and the true label of each fold is saved in `self.eval_result[fold]`

        Returns
        -----------
        estimated_pscore: array-like, shape (n_rounds_of_new_data, )
            Estimated propensity score, i.e., :math:`\\hat{\\pi}_b (a \\mid x)`.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=np.zeros_like(action),  # use dummy reward
            position=position,
            action_context=np.eye(self.n_actions, dtype=int),
        )
        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if n_folds == 1:
            self.fit(
                context=context,
                action=action,
                position=position,
            )
            return self.predict(context=context, action=action, position=position)
        else:
            estimated_pscore = np.zeros(context.shape[0])
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        if evaluate_model_performance:
            self.eval_result = {"y": [], "proba": []}
        for train_idx, test_idx in kf.split(context):
            self.fit(
                context=context[train_idx],
                action=action[train_idx],
                position=position[train_idx],
            )

            estimated_pscore[test_idx] = self.predict(
                context=context[test_idx],
                action=action[test_idx],
                position=position[test_idx],
            )
            if evaluate_model_performance:
                for pos_ in np.arange(self.len_list):
                    idx = position[test_idx] == pos_
                    if context[test_idx][idx].shape[0] == 0:
                        continue
                    proba_eval = self.base_model_list[pos_].predict_proba(
                        context[test_idx][idx]
                    )
                    self.eval_result["proba"].append(proba_eval)
                    self.eval_result["y"].append(action[test_idx][idx])
        return estimated_pscore
