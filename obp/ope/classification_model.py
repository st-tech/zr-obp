# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Classification Model Class for Estimating Propensity Score and Importance Weight."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
from sklearn.calibration import CalibratedClassifierCV

from ..utils import check_array, sample_action_fast, check_bandit_feedback_inputs


@dataclass
class ImportanceWeightEstimator(BaseEstimator):
    """Machine learning model to distinguish between the behavior and evaluation policy (:math:`\\Pr[C = 1 | x, a]`),
    where :math:`\\Pr[C=1|x,a]` is the probability that the data coming from the evaluation policy given action :math:`a` and :math:`x`.

    Parameters
    ------------
    base_model: BaseEstimator
        A machine learning model used to estimate the importance weights.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

    action_context: array-like, shape (n_actions, dim_action_context), default=None
        Context vector characterizing action (i.e., vector representation of each action).
        If not given, one-hot encoding of the action variable is used as default.
        If fitting_method is 'raw', action_context must be None.

    fitting_method: str, default='sample'
        Method to fit the classification model.
        Must be one of ['sample', 'raw']. Each method is defined as follows:
            - sample: actions are sampled by applying Gumbel-softmax trick to action_dist_at_position, and action features are represented by one-hot encoding of the sampled action.
            - raw: action_dist_at_position are directly encoded as action features.
        If fitting_method is 'raw', action_context must be None.

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
                f"fitting_method must be either 'sample' or 'raw', but {self.fitting_method} is given"
            )
        if self.fitting_method == "raw" and self.action_context is not None:
            check_array(array=self.action_context, name="action_context", expected_dim=2)
            if self.action_context.shape != (self.n_actions, self.n_actions) or not np.allclose(self.action_context, np.eye(self.n_actions)):
                raise ValueError(
                    "If fitting_method == 'raw', action_context must be None or identity matrix of size (n_actions, n_actions)."
                )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "base_model must be BaseEstimator or a child class of BaseEstimator"
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
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit the classification model on given logged bandit feedback data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is given, a classification model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.

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
        n_rounds = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            check_array(array=position, name="position", expected_dim=1)
            if position.max() >= self.len_list:
                raise ValueError(
                    f"position elements must be smaller than len_list, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
            raise ValueError(
                f"shape of action_dist must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
            )
        if not np.allclose(action_dist.sum(axis=1), 1):
            raise ValueError("action_dist must be a probability distribution")

        # If self.fitting_method != "sample", `sampled_action` has no information
        sampled_action = np.zeros(n_rounds, dtype=int)
        if self.fitting_method == "sample":
            for position_ in np.arange(self.len_list):
                idx = position == position_
                sampled_action_at_position = sample_action_fast(
                    action_dist=action_dist[idx][:, :, position_],
                    random_state=random_state,
                )
                sampled_action[idx] = sampled_action_at_position

        for position_ in np.arange(self.len_list):
            idx = position == position_
            action_dist_at_position = action_dist[idx][:, :, position_]
            X, y = self._pre_process_for_clf_model(
                context=context[idx],
                action=action[idx],
                action_dist_at_position=action_dist_at_position,
                sampled_action_at_position=sampled_action[idx],
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {position_}")
            self.base_model_list[position_].fit(X, y)

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
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds_of_new_data,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds_of_new_data,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is given, a classification model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.

        Returns
        ----------
        estimated_importance_weights: array-like, shape (n_rounds_of_new_data, )
            Importance weights estimated via supervised classification, i.e., :math:`\\hat{w}(x_t, a_t)`.

        """
        proba_of_evaluation_policy = np.zeros(action.shape[0])
        for position_ in np.arange(self.len_list):
            idx = position == position_
            X, _, = self._pre_process_for_clf_model(
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
        evaluate_model_performance: bool = False,
    ) -> np.ndarray:
        """Fit the classification model on given logged bandit feedback data and predict the importance weights of the same data.

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

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is given, a classification model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.

        n_folds: int, default=1
            Number of folds in the cross-fitting procedure.
            When 1 is given, the classification model is trained on the whole logged bandit feedback data.
            Please refer to https://arxiv.org/abs/2002.08536 about the details of the cross-fitting procedure.

        random_state: int, default=None
            `random_state` affects the ordering of the indices, which controls the randomness of each fold.
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html for the details.

        evaluate_model_performance: bool, default=False
            Whether the performance of the classification model is evaluated or not.
            When True is given, the predicted probability of the classification model and the true label of each fold is saved in `self.eval_result[fold]`

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
        check_array(array=action_dist, name="action_dist", expected_dim=3)

        n_rounds = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"position elements must be smaller than len_list, but the maximum value is {position.max()} (>= {self.len_list})"
                )

        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

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
                random_state=random_state,
            )
            return self.predict(context=context, action=action, position=position)
        else:
            estimated_importance_weights = np.zeros(n_rounds)
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
                    for position_ in np.arange(self.len_list):
                        idx = position[test_idx] == position_
                        sampled_action_at_position = sample_action_fast(
                            action_dist=action_dist[test_idx][idx][:, :, position_],
                            random_state=random_state,
                        )
                        sampled_action[idx] = sampled_action_at_position
                for position_ in np.arange(self.len_list):
                    idx = position[test_idx] == position_
                    action_dist_at_position = action_dist[test_idx][idx][
                        :, :, position_
                    ]
                    X, y = self._pre_process_for_clf_model(
                        context=context[test_idx][idx],
                        action=action[test_idx][idx],
                        action_dist_at_position=action_dist_at_position,
                        sampled_action_at_position=sampled_action[idx],
                    )
                    proba_of_evaluation_policy = self.base_model_list[
                        position_
                    ].predict_proba(X)[:, 1]
                    self.eval_result["proba"].append(proba_of_evaluation_policy)
                    self.eval_result["y"].append(y)
        return estimated_importance_weights

    def _pre_process_for_clf_model(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_dist_at_position: Optional[np.ndarray] = None,
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
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        action_context: array-like, shape shape (n_actions, dim_action_context)
            Context vector characterizing action (i.e., vector representation of each action).

        action_dist_at_position: array-like, shape (n_rounds, n_actions,)
            Action choice probabilities of evaluation policy of each position (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

        sampled_action_at_position: array-like, shape (n_rounds, n_actions,)
            One-hot encoding of actions sampled by evaluation policy of each position.

        """
        behavior_policy_feature = np.c_[context, self.action_context[action]]
        if is_prediction:
            return behavior_policy_feature, None
        if self.fitting_method == "raw":
            evaluation_policy_feature = np.c_[context, action_dist_at_position]
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
        A machine learning model used to estimate the mean reward function.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.

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
    calibration_cv: int = 2

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        check_scalar(self.calibration_cv, "calibration_cv", int)
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "base_model must be BaseEstimator or a child class of BaseEstimator"
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
        """Fit the classification model on given logged bandit feedback data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is given, a classification model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.

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
                    f"position elements must be smaller than len_list, but the maximum value is {position.max()} (>= {self.len_list})"
                )

        for position_ in np.arange(self.len_list):
            idx = position == position_
            if context[idx].shape[0] == 0:
                raise ValueError(f"No training data at position {position_}")
            self.base_model_list[position_].fit(context[idx], action[idx])

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
            Context vectors observed in each round of the logged bandit feedback, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds_of_new_data,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        position: array-like, shape (n_rounds_of_new_data,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is given, a classification model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.

        Returns
        ----------
        estimated_pscore: array-like, shape (n_rounds_of_new_data, )
            Estimated propensity scores, i.e., :math:`\\hat{\\pi}_b (a \\mid x)`.

        """
        estimated_pscore = np.zeros(action.shape[0])
        for position_ in np.arange(self.len_list):
            idx = position == position_
            if context[idx].shape[0] == 0:
                continue
            estimated_pscore[idx] = self.base_model_list[position_].predict_proba(
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
        """Fit the classification model on given logged bandit feedback data and predict the propensity score of the same data.

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

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit data.
            If None is given, a classification model assumes that there is only one position.
            When `len_list` > 1, this position argument has to be set.

        n_folds: int, default=1
            Number of folds in the cross-fitting procedure.
            When 1 is given, the classification model is trained on the whole logged bandit feedback data.
            Please refer to https://arxiv.org/abs/2002.08536 about the details of the cross-fitting procedure.

        random_state: int, default=None
            `random_state` affects the ordering of the indices, which controls the randomness of each fold.
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html for the details.

        evaluate_model_performance: bool, default=False
            Whether the performance of the classification model is evaluated or not.
            When True is given, the predicted probability of the classification model and the true label of each fold is saved in `self.eval_result[fold]`

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

        n_rounds = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"position elements must be smaller than len_list, but the maximum value is {position.max()} (>= {self.len_list})"
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
            estimated_pscore = np.zeros(n_rounds)
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
                for position_ in np.arange(self.len_list):
                    idx = position[test_idx] == position_
                    if context[test_idx][idx].shape[0] == 0:
                        continue
                    proba_eval = self.base_model_list[position_].predict_proba(
                        context[test_idx][idx]
                    )
                    self.eval_result["proba"].append(proba_eval)
                    self.eval_result["y"].append(action[test_idx][idx])
        return estimated_pscore
