# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Offline Bandit Algorithms."""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch.optim as optim
from tqdm import tqdm

from obp.ope import RegressionModel

from ..utils import check_array
from ..utils import check_bandit_feedback_inputs
from ..utils import check_tensor
from ..utils import softmax as softmax_axis1
from .base import BaseOfflinePolicyLearner


@dataclass
class IPWLearner(BaseOfflinePolicyLearner):
    """Off-policy learner based on Inverse Probability Weighting and Supervised Classification.

    Parameters
    -----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    base_classifier: ClassifierMixin
        Machine learning classifier used to train an offline decision making policy.

    References
    ------------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Damien Lefortier, Adith Swaminathan, Xiaotao Gu, Thorsten Joachims, and Maarten de Rijke.
    "Large-scale Validation of Counterfactual Learning Methods: A Test-Bed.", 2016.

    """

    base_classifier: Optional[ClassifierMixin] = None

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()
        if self.base_classifier is None:
            self.base_classifier = LogisticRegression(random_state=12345)
        else:
            if not is_classifier(self.base_classifier):
                raise ValueError("`base_classifier` must be a classifier")
        self.base_classifier_list = [
            clone(self.base_classifier) for _ in np.arange(self.len_list)
        ]
        self.policy_name = "IPWLearner"

    @staticmethod
    def _create_train_data_for_opl(
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create training data for off-policy learning.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        Returns
        --------
        (X, sample_weight, y): Tuple[np.ndarray, np.ndarray, np.ndarray]
            Feature vectors, sample weights, and outcome for training the base machine learning model.

        """
        return context, (reward / pscore), action

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> None:
        """Fits an offline bandit policy on the given logged bandit data.

        Note
        --------
        This `fit` method trains a deterministic policy :math:`\\pi: \\mathcal{X} \\rightarrow \\mathcal{A}`
        via a cost-sensitive classification reduction as follows:

        .. math::

            \\hat{\\pi}
            & \\in \\arg \\max_{\\pi \\in \\Pi} \\hat{V}_{\\mathrm{IPW}} (\\pi ; \\mathcal{D}) \\\\
            & = \\arg \\max_{\\pi \\in \\Pi} \\mathbb{E}_{n} \\left[\\frac{\\mathbb{I} \\{\\pi (x_{i})=a_{i} \\}}{\\pi_{b}(a_{i} | x_{i})} r_{i} \\right] \\\\
            & = \\arg \\min_{\\pi \\in \\Pi} \\mathbb{E}_{n} \\left[\\frac{r_i}{\\pi_{b}(a_{i} | x_{i})} \\mathbb{I} \\{\\pi (x_{i}) \\neq a_{i} \\} \\right],

        where :math:`\\mathbb{E}_{n} [\cdot]` is the empirical average over observations in :math:`\\mathcal{D}`.
        See the reference for the details.


        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a learner assumes that only a single action is chosen for each data.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )
        if (reward < 0).any():
            raise ValueError(
                "A negative value is found in `reward`."
                "`obp.policy.IPWLearner` cannot handle negative rewards,"
                "and please use `obp.policy.NNPolicyLearner` instead."
            )
        if pscore is None:
            n_actions = np.int32(action.max() + 1)
            pscore = np.ones_like(action) / n_actions
        if self.len_list == 1:
            position = np.zeros_like(action, dtype=int)
        else:
            if position is None:
                raise ValueError("When `self.len_list > 1`, `position` must be given.")

        for p in np.arange(self.len_list):
            X, sample_weight, y = self._create_train_data_for_opl(
                context=context[position == p],
                action=action[position == p],
                reward=reward[position == p],
                pscore=pscore[position == p],
            )
            self.base_classifier_list[p].fit(X=X, y=y, sample_weight=sample_weight)

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict best actions for new data.

        Note
        --------
        Action set predicted by this `predict` method can contain duplicate items.
        If a non-repetitive action set is needed, please use the `sample_action` method.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        action_dist: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choices made by a classifier, which can contain duplicate items.
            If a non-repetitive action set is needed, please use the `sample_action` method.

        """
        check_array(array=context, name="context", expected_dim=2)

        n_rounds = context.shape[0]
        action_dist = np.zeros((n_rounds, self.n_actions, self.len_list))
        for p in np.arange(self.len_list):
            predicted_actions_at_position = self.base_classifier_list[p].predict(
                context
            )
            action_dist[
                np.arange(n_rounds),
                predicted_actions_at_position,
                np.ones(n_rounds, dtype=int) * p,
            ] += 1
        return action_dist

    def predict_score(self, context: np.ndarray) -> np.ndarray:
        """Predict non-negative scores for all possible pairs of actions and positions.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        score_predicted: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Scores for all possible pairs of actions and positions predicted by a classifier.

        """
        check_array(array=context, name="context", expected_dim=2)

        n = context.shape[0]
        score_predicted = np.zeros((n, self.n_actions, self.len_list))
        for p in np.arange(self.len_list):
            score_predicteds_at_position = self.base_classifier_list[p].predict_proba(
                context
            )
            score_predicted[:, :, p] = score_predicteds_at_position
        return score_predicted

    def sample_action(
        self,
        context: np.ndarray,
        tau: Union[int, float] = 1.0,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Sample a ranking of (non-repetitive) actions from the Plackett-Luce ranking distribution.

        Note
        --------
        This `sample_action` method samples a **non-repetitive** ranking of actions for new data
        :math:`x \\in \\mathcal{X}` via the so-called "Gumbel Softmax trick" as follows.

        .. math::

            \\s (x,a) = \\hat{f}(x,a) / \\tau + \\gamma_{x,a}, \\quad \\gamma_{x,a} \\sim \\mathrm{Gumbel}(0,1)

        :math:`\\tau` is a temperature hyperparameter.
        :math:`f: \\mathcal{X} \\times \\mathcal{A} \\times \\mathcal{K} \\rightarrow \\mathbb{R}_{+}`
        is a scoring function which is now implemented in the `predict_score` method.
        When `len_list > 0`,  the expected rewards estimated at different positions will be averaged to form :math:`f(x,a)`.
        :math:`\\gamma_{x,a}` is a random variable sampled from the Gumbel distribution.
        By sorting the actions based on :math:`\\s (x,a)` for each context, we can efficiently sample a ranking from
        the Plackett-Luce ranking distribution.

        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        tau: int or float, default=1.0
            A temperature parameter that controls the randomness of the action choice
            by scaling the scores before applying softmax.
            As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

        random_state: int, default=None
            Controls the random seed in sampling actions.

        Returns
        -----------
        sampled_ranking: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Ranking of actions sampled via the Gumbel softmax trick.

        """
        check_array(array=context, name="context", expected_dim=2)
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        n = context.shape[0]
        random_ = check_random_state(random_state)
        sampled_ranking = np.zeros((n, self.n_actions, self.len_list))
        scores = self.predict_score(context=context).mean(2) / tau
        scores += random_.gumbel(size=scores.shape)
        sampled_ranking_full = np.argsort(-scores, axis=1)
        for p in np.arange(self.len_list):
            sampled_ranking[np.arange(n), sampled_ranking_full[:, p], p] = 1
        return sampled_ranking

    def predict_proba(
        self,
        context: np.ndarray,
        tau: Union[int, float] = 1.0,
    ) -> np.ndarray:
        """Obtains action choice probabilities for new data based on scores predicted by a classifier.

        Note
        --------
        This `predict_proba` method obtains action choice probabilities for new data :math:`x \\in \\mathcal{X}`
        by applying the softmax function as follows:

        .. math::

            P (A = a | x) = \\frac{\\mathrm{exp}(f(x,a) / \\tau)}{\\sum_{a^{\\prime} \\in \\mathcal{A}} \\mathrm{exp}(f(x,a^{\\prime}) / \\tau)},

        where :math:`A` is a random variable representing an action, and :math:`\\tau` is a temperature hyperparameter.
        :math:`f: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}_{+}`
        is a scoring function which is now implemented in the `predict_score` method.

        **Note that this method can be used only when `len_list=1`, please use the `sample_action` method otherwise.**

        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        tau: int or float, default=1.0
            A temperature parameter that controls the randomness of the action choice
            by scaling the scores before applying softmax.
            As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

        Returns
        -----------
        choice_prob: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choice probabilities obtained by a trained classifier.

        """
        assert (
            self.len_list == 1
        ), "predict_proba method cannot be used when `len_list != 1`"
        check_array(array=context, name="context", expected_dim=2)
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        score_predicted = self.predict_score(context=context)
        choice_prob = softmax(score_predicted / tau, axis=1)
        return choice_prob


@dataclass
class QLearner(BaseOfflinePolicyLearner):
    """Off-policy learner based on Direct Method (Reward Regression).

    Parameters
    -----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    base_model: BaseEstimator
        Machine learning model used to estimate the q function (expected reward function).

    fitting_method: str, default='normal'
        Method to fit the regression model.
        Must be one of ['normal', 'iw'] where 'iw' stands for importance weighting.

    """

    base_model: Optional[BaseEstimator] = None
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

        self.q_estimator = RegressionModel(
            n_actions=self.n_actions,
            len_list=self.len_list,
            base_model=self.base_model,
            fitting_method=self.fitting_method,
        )

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> None:
        """Fits an offline bandit policy on the given logged bandit feedback data.

        Note
        --------
        This `fit` method trains an estimator for the q function :math:`\\q(x,a) := \\mathbb{E} [r \\mid x, a]` as follows.

        .. math::

            \\hat{\\q} \\in \\arg \\min_{\\q \\in \\Q} \\mathbb{E}_{n} [ \\ell ( r_i, q (x_i,a_i) )  ]

        where :math:`\\ell` is a loss function in training the q estimator.


        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a learner assumes that only a single action is chosen for each data.
            When `len_list` > 1, position has to be set.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )
        if pscore is None:
            n_actions = np.int32(action.max() + 1)
            pscore = np.ones_like(action) / n_actions
        if self.len_list == 1:
            position = np.zeros_like(action, dtype=int)
        else:
            if position is None:
                raise ValueError("When `self.len_list > 1`, `position` must be given.")

        unif_action_dist = np.ones((context.shape[0], self.n_actions, self.len_list))
        self.q_estimator.fit(
            context=context,
            action=action,
            reward=reward,
            position=position,
            pscore=pscore,
            action_dist=unif_action_dist,
        )

    def predict(
        self,
        context: np.ndarray,
        tau: Union[int, float] = 1.0,
    ) -> np.ndarray:
        """Predict best actions for new data deterministically.

        Note
        --------
        This `predict` method predicts the best actions for new data deterministically as follows.

        .. math::

            \\hat{a}_i \\in \\arg \\max_{a \\in \\mathcal{A}} \\hat{q}(x_i, a)

        where :math:`\\hat{q}(x,a)` is an estimator for the q function :math:`\\q(x,a) := \\mathbb{E} [r \\mid x, a]`.
        Note that action sets predicted by this `predict` method can contain duplicate items.
        If a non-repetitive action set is needed, please use the `sample_action` method.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        action_dist: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Deterministic action choices made by the QLearner.
            The output can contain duplicated items (when `len_list > 1`).

        """
        check_array(array=context, name="context", expected_dim=2)
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        q_hat = self.predict_score(context=context)
        q_hat_argmax = np.argmax(q_hat, axis=1).astype(int)

        n = context.shape[0]
        action_dist = np.zeros_like(q_hat)
        for p in np.arange(self.len_list):
            action_dist[np.arange(n), q_hat_argmax[:, p], p] = 1
        return action_dist

    def predict_score(self, context: np.ndarray) -> np.ndarray:
        """Predict the expected rewards for all possible pairs of actions and positions.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        q_hat: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Expected rewards for all possible pairs of actions and positions. :math:`\\hat{q}(x,a)`.

        """
        check_array(array=context, name="context", expected_dim=2)

        q_hat = self.q_estimator.predict(context=context)
        return q_hat

    def sample_action(
        self,
        context: np.ndarray,
        tau: Union[int, float] = 1.0,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Sample a ranking of (non-repetitive) actions from the Plackett-Luce ranking distribution.

        Note
        --------
        This `sample_action` method samples a ranking of (non-repetitive) actions for new data
        based on :math:`\\hat{q}` and the so-called "Gumbel Softmax trick" as follows.

        .. math::

            \\s (x,a) = \\hat{q}(x,a) / \\tau + \\gamma_{x,a}, \\quad \\gamma_{x,a} \\sim \\mathrm{Gumbel}(0,1)

        :math:`\\tau` is a temperature hyperparameter.
        :math:`\\hat{q}: \\mathcal{X} \\times \\mathcal{A} \\times \\mathcal{K} \\rightarrow \\mathbb{R}_{+}`
        is a q function estimator, which is now implemented in the `predict_score` method.
        When `len_list > 0`,  the expected rewards estimated at different positions will be averaged to form :math:`f(x,a)`.
        :math:`\\gamma_{x,a}` is a random variable sampled from the Gumbel distribution.
        By sorting the actions based on :math:`\\s (x,a)` for each context, we can efficiently sample a ranking from
        the Plackett-Luce ranking distribution.

        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        tau: int or float, default=1.0
            A temperature parameter that controls the randomness of the action choice
            by scaling the scores before applying softmax.
            As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

        random_state: int, default=None
            Controls the random seed in sampling actions.

        Returns
        -----------
        sampled_action: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Ranking of actions sampled from the Plackett-Luce ranking distribution via the Gumbel softmax trick.

        """
        check_array(array=context, name="context", expected_dim=2)
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        n = context.shape[0]
        random_ = check_random_state(random_state)
        sampled_action = np.zeros((n, self.n_actions, self.len_list))
        scores = self.predict_score(context=context).mean(2) / tau
        scores += random_.gumbel(size=scores.shape)
        ranking = np.argsort(-scores, axis=1)
        for p in np.arange(self.len_list):
            sampled_action[np.arange(n), ranking[:, p], p] = 1
        return sampled_action

    def predict_proba(
        self,
        context: np.ndarray,
        tau: Union[int, float] = 1.0,
    ) -> np.ndarray:
        """Obtains action choice probabilities for new data based on the estimated expected rewards.

        Note
        --------
        This `predict_proba` method obtains action choice probabilities for new data based on :math:`\\hat{q}` as follows.

        .. math::

            \\pi_{l} (a|x) = \\frac{\\mathrm{exp}( \\hat{q}_{l}(x,a) / \\tau)}{\\sum_{a^{\\prime} \\in \\mathcal{A}} \\mathrm{exp}( \\hat{q}_{l}(x,a^{\\prime}) / \\tau)}

        where :math:`\\pi_{l} (a|x)` is the resulting action choice probabilities at position :math:`l`.
        :math:`\\tau` is a temperature hyperparameter.
        :math:`\\hat{q}: \\mathcal{X} \\times \\mathcal{A} \\times \\mathcal{K} \\rightarrow \\mathbb{R}_{+}`
        is a q function estimator for position :math:`l`, which is now implemented in the `predict_score` method.

        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        tau: int or float, default=1.0
            A temperature parameter that controls the randomness of the action choice
            by scaling the scores before applying softmax.
            As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

        Returns
        -----------
        action_dist: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choice probabilities obtained from the estimated expected rewards.

        """
        check_array(array=context, name="context", expected_dim=2)
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        q_hat = self.predict_score(context=context)
        action_dist = softmax_axis1(q_hat / tau)
        return action_dist


@dataclass
class NNPolicyLearner(BaseOfflinePolicyLearner):
    """Off-policy learner parameterized by a neural network.

    Parameters
    -----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    dim_context: int
        Number of dimensions of context vectors.

    off_policy_objective: str
        An OPE estimator used to estimate the policy gradient.
        Must be one of 'dm', 'ipw', 'dr', 'snipw', 'ipw-os', and 'ipw-subgauss'.
        They stand for
            - Direct Method
            - Inverse Probability Weighting
            - Doubly Robust
            - Self-Normalized Inverse Probability Weighting
            - Inverse Probability Weighting with Optimistic Shrinkage
            - Inverse Probability Weighting with Sungaussian Weight
        , respectively.

    lambda_: float, default=np.inf
        A hyperparameter used for 'snipw', 'ipw-os', and 'ipw-subgauss'.
        When `off_policy_objective`='snipw', `lambda_` is used to shift the reward.
        Otherwise, `lambda_` is used to modify or shrinkage the importance weight.

    policy_reg_param: float, default=0.0
        A hypeparameter to control the policy regularization. :math:`\\lambda_{pol}`.

    var_reg_param: float, default=0.0
        A hypeparameter to control the variance regularization. :math:`\\lambda_{var}`.

    hidden_layer_size: Tuple[int, ...], default = (100,)
        The i-th element specifies the size of the i-th layer.

    activation: str, default='relu'
        Activation function.
        Must be one of the followings:

        - 'identity', the identity function, :math:`f(x) = x`.
        - 'logistic', the sigmoid function, :math:`f(x) = \\frac{1}{1 + \\exp(x)}`.
        - 'tanh', the hyperbolic tangent function, `:math:f(x) = \\frac{\\exp(x) - \\exp(-x)}{\\exp(x) + \\exp(-x)}`
        - 'relu', the rectified linear unit function, `:math:f(x) = \\max(0, x)`

    solver: str, default='adam'
        Optimizer of the neural network.
        Must be one of the followings:

        - 'sgd', Stochastic Gradient Descent.
        - 'adam', Adam (Kingma and Ba 2014).
        - 'adagrad', Adagrad (Duchi et al. 2011).

    alpha: float, default=0.001
        L2 penalty.

    batch_size: Union[int, str], default="auto"
        Batch size for SGD, Adagrad, and Adam.
        If "auto", the maximum of 200 and the number of samples is used.
        If integer, must be positive.

    learning_rate_init: int, default=0.0001
        Initial learning rate for SGD, Adagrad, and Adam.

    max_iter: int, default=200
        Number of epochs for SGD, Adagrad, and Adam.

    shuffle: bool, default=True
        Whether to shuffle samples in SGD and Adam.

    random_state: Optional[int], default=None
        Controls the random seed.

    tol: float, default=1e-4
        Tolerance for training.
        When the training loss is not improved at least `tol' for `n_iter_no_change' consecutive iterations,
        training is stopped.

    momentum: float, default=0.9
        Momentum for SGD.
        Must be in the range of [0., 1.].

    nesterovs_momentum: bool, default=True
        Whether to use Nesterovs momentum.

    early_stopping: bool, default=False
        Whether to use early stopping for SGD, Adagrad, and Adam.
        If set to true, `validation_fraction' of training data is used as validation data,
        and training is stopped when the validation loss is not improved at least `tol' for `n_iter_no_change' consecutive iterations.

    validation_fraction: float, default=0.1
        Fraction of validation data when early stopping is used.
        Must be in the range of (0., 1.].

    beta_1: float, default=0.9
        Coefficient used for computing running average of gradient for Adam.
        Must be in the range of [0., 1.].

    beta_2: float, default=0.999
        Coefficient used for computing running average of the square of gradient for Adam.
        Must be in the range of [0., 1.].

    epsilon: float, default=1e-8
        Term for numerical stability in Adam.

    n_iter_no_change: int, default=10
        Maximum number of not improving epochs when early stopping is used.

    q_func_estimator_hyperparams: Dict, default=None
        A set of hyperparameters to define q function estimator.

    References:
    ------------
    Dong .C. Liu and Jorge Nocedal.
    "On the Limited Memory Method for Large Scale Optimization.", 1989

    Diederik P. Kingma and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization.", 2014

    John Duchi, Elad Hazan, and Yoram Singer.
    "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.", 2011.

    Thorsten Joachims, Adith Swaminathan, and Maarten de Rijke.
    ""Deep Learning for Logged Bandit Feedback."", 2018.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    Alberto Maria Metelli, Alessio Russo, and Marcello Restelli.
    "Subgaussian and Differentiable Importance Sampling for Off-Policy Evaluation and Learning.", 2021.

    """

    dim_context: Optional[int] = None
    off_policy_objective: Optional[str] = None
    lambda_: Optional[float] = None
    policy_reg_param: float = 0.0
    var_reg_param: float = 0.0
    hidden_layer_size: Tuple[int, ...] = (100,)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: Union[int, str] = "auto"
    learning_rate_init: float = 0.0001
    max_iter: int = 200
    shuffle: bool = True
    random_state: Optional[int] = None
    tol: float = 1e-4
    momentum: float = 0.9
    nesterovs_momentum: bool = True
    early_stopping: bool = False
    validation_fraction: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    n_iter_no_change: int = 10
    q_func_estimator_hyperparams: Optional[Dict] = None

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

        check_scalar(self.dim_context, "dim_context", int, min_val=1)

        if self.off_policy_objective not in [
            "dm",
            "ipw",
            "dr",
            "snipw",
            "ipw-os",
            "ipw-subgauss",
        ]:
            raise ValueError(
                "`off_policy_objective` must be one of 'dm', 'ipw', 'dr', 'snipw', 'ipw-os', 'ipw-subgauss'"
                f", but {self.off_policy_objective} is given"
            )

        if self.off_policy_objective == "ipw-subgauss":
            if self.lambda_ is None:
                self.lambda_ = 0.001
            check_scalar(
                self.lambda_,
                "lambda_",
                (int, float),
                min_val=0.0,
                max_val=1.0,
            )

        elif self.off_policy_objective == "snipw":
            if self.lambda_ is None:
                self.lambda_ = 0.0
            check_scalar(
                self.lambda_,
                "lambda_",
                (int, float),
                min_val=0.0,
            )

        elif self.off_policy_objective == "ipw-os":
            if self.lambda_ is None:
                self.lambda_ = 10000
            check_scalar(
                self.lambda_,
                "lambda_",
                (int, float),
                min_val=0.0,
            )

        check_scalar(
            self.policy_reg_param,
            "policy_reg_param",
            (int, float),
            min_val=0.0,
        )

        check_scalar(
            self.var_reg_param,
            "var_reg_param",
            (int, float),
            min_val=0.0,
        )

        if not isinstance(self.hidden_layer_size, tuple) or any(
            [not isinstance(h, int) or h <= 0 for h in self.hidden_layer_size]
        ):
            raise ValueError(
                f"`hidden_layer_size` must be a tuple of positive integers, but {self.hidden_layer_size} is given"
            )

        if self.solver not in ("adagrad", "sgd", "adam"):
            raise ValueError(
                f"`solver` must be one of 'adam', 'adagrad', or 'sgd', but {self.solver} is given"
            )

        check_scalar(self.alpha, "alpha", float, min_val=0.0)

        if self.batch_size != "auto" and (
            not isinstance(self.batch_size, int) or self.batch_size <= 0
        ):
            raise ValueError(
                f"`batch_size` must be a positive integer or 'auto', but {self.batch_size} is given"
            )

        check_scalar(self.learning_rate_init, "learning_rate_init", float)
        if self.learning_rate_init <= 0.0:
            raise ValueError(
                f"`learning_rate_init`= {self.learning_rate_init}, must be > 0.0"
            )

        check_scalar(self.max_iter, "max_iter", int, min_val=1)

        if not isinstance(self.shuffle, bool):
            raise ValueError(f"`shuffle` must be a bool, but {self.shuffle} is given")

        check_scalar(self.tol, "tol", float)
        if self.tol <= 0.0:
            raise ValueError(f"`tol`= {self.tol}, must be > 0.0")

        check_scalar(self.momentum, "momentum", float, min_val=0.0, max_val=1.0)

        if not isinstance(self.nesterovs_momentum, bool):
            raise ValueError(
                f"`nesterovs_momentum` must be a bool, but {self.nesterovs_momentum} is given"
            )

        if not isinstance(self.early_stopping, bool):
            raise ValueError(
                f"`early_stopping` must be a bool, but {self.early_stopping} is given"
            )

        check_scalar(
            self.validation_fraction, "validation_fraction", float, max_val=1.0
        )
        if self.validation_fraction <= 0.0:
            raise ValueError(
                f"`validation_fraction`= {self.validation_fraction}, must be > 0.0"
            )

        if self.q_func_estimator_hyperparams is not None:
            if not isinstance(self.q_func_estimator_hyperparams, dict):
                raise ValueError(
                    "`q_func_estimator_hyperparams` must be a dict"
                    f", but {type(self.q_func_estimator_hyperparams)} is given"
                )
        check_scalar(self.beta_1, "beta_1", float, min_val=0.0, max_val=1.0)
        check_scalar(self.beta_2, "beta_2", float, min_val=0.0, max_val=1.0)
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0)
        check_scalar(self.n_iter_no_change, "n_iter_no_change", int, min_val=1)

        if self.random_state is not None:
            self.random_ = check_random_state(self.random_state)
            torch.manual_seed(self.random_state)

        if self.activation == "identity":
            activation_layer = nn.Identity
        elif self.activation == "logistic":
            activation_layer = nn.Sigmoid
        elif self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU
        else:
            raise ValueError(
                "`activation` must be one of 'identity', 'logistic', 'tanh', 'relu', or 'elu'"
                f", but {self.activation} is given"
            )

        layer_list = []
        input_size = self.dim_context

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.n_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        if self.off_policy_objective in ["dr", "dm"]:
            if self.q_func_estimator_hyperparams is not None:
                self.q_func_estimator_hyperparams["n_actions"] = self.n_actions
                self.q_func_estimator_hyperparams["dim_context"] = self.dim_context
                self.q_func_estimator = QFuncEstimator(
                    **self.q_func_estimator_hyperparams
                )
            else:
                self.q_func_estimator = QFuncEstimator(
                    n_actions=self.n_actions, dim_context=self.dim_context
                )

    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        position: np.ndarray,
        **kwargs,
    ) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
        """Create training data for off-policy learning.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a learner assumes that only a single action is chosen for each data.

        Returns
        --------
        (training_data_loader, validation_data_loader): Tuple[DataLoader, Optional[DataLoader]]
            Training and validation data loaders in PyTorch

        """
        if self.batch_size == "auto":
            batch_size_ = min(200, context.shape[0])
        else:
            check_scalar(self.batch_size, "batch_size", int, min_val=1)
            batch_size_ = self.batch_size

        dataset = NNPolicyDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(action).long(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(position).float(),
        )

        if self.early_stopping:
            if context.shape[0] <= 1:
                raise ValueError(
                    f"the number of samples is too small ({context.shape[0]}) to create validation data"
                )

            validation_size = max(int(context.shape[0] * self.validation_fraction), 1)
            training_size = context.shape[0] - validation_size
            training_dataset, validation_dataset = torch.utils.data.random_split(
                dataset, [training_size, validation_size]
            )
            training_data_loader = torch.utils.data.DataLoader(
                training_dataset,
                batch_size=batch_size_,
                shuffle=self.shuffle,
            )
            validation_data_loader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=batch_size_,
                shuffle=self.shuffle,
            )

            return training_data_loader, validation_data_loader

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_,
            shuffle=self.shuffle,
        )

        return data_loader, None

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> None:
        """Fits an offline bandit policy on the given logged bandit data.

        Note
        ----------
        Given the training data :math:`\\mathcal{D}`, this policy maximizes the following objective function:

        .. math::

            \\hat{V}(\\pi_\\theta; \\mathcal{D}) - \\alpha \\Omega(\\theta)

        where :math:`\\hat{V}` is an OPE estimator and :math:`\\alpha \\Omega(\\theta)` is a regularization term.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a learner assumes that only a single action is chosen for each data.
            When `len_list` > 1, position has to be set.
            Currently, this feature is not supported.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
        )
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions
        if self.len_list == 1:
            position = np.zeros_like(action, dtype=int)

        # train q function estimator when it is needed to train NNPolicy
        if self.off_policy_objective in ["dr", "dm"]:
            self.q_func_estimator.fit(
                context=context,
                action=action,
                reward=reward,
            )
        if self.solver == "sgd":
            optimizer = optim.SGD(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                momentum=self.momentum,
                weight_decay=self.alpha,
                nesterov=self.nesterovs_momentum,
            )
        elif self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                eps=self.epsilon,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.Adam(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                betas=(self.beta_1, self.beta_2),
                eps=self.epsilon,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError(
                "`solver` must be one of 'adam', 'adagrad', or 'sgd'"
            )

        training_data_loader, validation_data_loader = self._create_train_data_for_opl(
            context, action, reward, pscore, position
        )

        # start policy training
        n_not_improving_training = 0
        previous_training_loss = None
        n_not_improving_validation = 0
        previous_validation_loss = None
        for _ in tqdm(np.arange(self.max_iter), desc="policy learning"):
            self.nn_model.train()
            for x, a, r, p, pos in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x).unsqueeze(-1)
                policy_grad_arr = self._estimate_policy_gradient(
                    context=x,
                    reward=r,
                    action=a,
                    pscore=p,
                    action_dist=pi,
                    position=pos,
                )
                policy_constraint = self._estimate_policy_constraint(
                    action=a,
                    pscore=p,
                    action_dist=pi,
                )
                loss = -policy_grad_arr.mean()
                loss += self.policy_reg_param * policy_constraint
                loss += self.var_reg_param * torch.var(policy_grad_arr)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                if previous_training_loss is not None:
                    if loss_value - previous_training_loss < self.tol:
                        n_not_improving_training += 1
                    else:
                        n_not_improving_training = 0
                if n_not_improving_training >= self.n_iter_no_change:
                    break
                previous_training_loss = loss_value

            if self.early_stopping:
                self.nn_model.eval()
                for x, a, r, p, pos in validation_data_loader:
                    pi = self.nn_model(x).unsqueeze(-1)
                    policy_grad_arr = self._estimate_policy_gradient(
                        context=x,
                        reward=r,
                        action=a,
                        pscore=p,
                        action_dist=pi,
                        position=pos,
                    )
                    policy_constraint = self._estimate_policy_constraint(
                        action=a,
                        pscore=p,
                        action_dist=pi,
                    )
                    loss = -policy_grad_arr.mean()
                    loss += self.policy_reg_param * policy_constraint
                    loss += self.var_reg_param * torch.var(policy_grad_arr)
                    loss_value = loss.item()
                    if previous_validation_loss is not None:
                        if loss_value - previous_validation_loss < self.tol:
                            n_not_improving_validation += 1
                        else:
                            n_not_improving_validation = 0
                    if n_not_improving_validation > self.n_iter_no_change:
                        break
                    previous_validation_loss = loss_value

    def _estimate_policy_gradient(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        pscore: torch.Tensor,
        action_dist: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the policy gradient.

        Parameters
        -----------
        context: array-like, shape (batch_size, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, shape (batch_size,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (batch_size,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (batch_size,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (batch_size, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        Returns
        ----------
        estimated_policy_grad_arr: array-like, shape (batch_size,)
            Rewards of each data estimated by an OPE estimator.

        """
        current_pi = action_dist[:, :, 0].detach()
        log_prob = torch.log(action_dist[:, :, 0])
        idx_tensor = torch.arange(action.shape[0], dtype=torch.long)

        if self.off_policy_objective == "dm":
            q_hat = self.q_func_estimator.predict(
                context=context,
            )
            estimated_policy_grad_arr = torch.sum(q_hat * current_pi * log_prob, dim=1)

        elif self.off_policy_objective == "ipw":
            iw = current_pi[idx_tensor, action] / pscore
            estimated_policy_grad_arr = iw * reward
            estimated_policy_grad_arr *= log_prob[idx_tensor, action]

        elif self.off_policy_objective == "dr":
            q_hat = self.q_func_estimator.predict(
                context=context,
            )
            q_hat_factual = q_hat[idx_tensor, action]
            iw = current_pi[idx_tensor, action] / pscore
            estimated_policy_grad_arr = iw * (reward - q_hat_factual)
            estimated_policy_grad_arr *= log_prob[idx_tensor, action]
            estimated_policy_grad_arr += torch.sum(q_hat * current_pi * log_prob, dim=1)

        elif self.off_policy_objective == "snipw":
            iw = current_pi[idx_tensor, action] / pscore
            estimated_policy_grad_arr = iw * (reward - self.lambda_)
            estimated_policy_grad_arr *= log_prob[idx_tensor, action]

        elif self.off_policy_objective == "ipw-os":
            iw = current_pi[idx_tensor, action] / pscore
            iw_ = (self.lambda_ - (iw**2)) / ((iw**2 + self.lambda_) ** 2)
            iw_ *= self.lambda_ * iw
            estimated_policy_grad_arr = iw_ * reward
            estimated_policy_grad_arr *= log_prob[idx_tensor, action]

        elif self.off_policy_objective == "ipw-subgauss":
            iw = current_pi[idx_tensor, action] / pscore
            iw_ = (1 - self.lambda_) * iw
            iw_ /= (1 - self.lambda_ + self.lambda_ * iw) ** 2
            estimated_policy_grad_arr = iw_ * reward
            estimated_policy_grad_arr *= log_prob[idx_tensor, action]

        return estimated_policy_grad_arr

    def _estimate_policy_constraint(
        self,
        action: torch.Tensor,
        pscore: torch.Tensor,
        action_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the policy constraint term.

        Parameters
        -----------
        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        """
        idx_tensor = torch.arange(action.shape[0], dtype=torch.long)
        iw = action_dist[idx_tensor, action, 0] / pscore

        return torch.log(iw.mean())

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict best actions for new data.

        Note
        --------
        Action set predicted by this `predict` method can contain duplicate items.
        If a non-repetitive action set is needed, please use the `sample_action` method.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        action_dist: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choices made by a classifier, which can contain duplicate items.
            If a non-repetitive action set is needed, please use the `sample_action` method.

        """
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )

        self.nn_model.eval()
        x = torch.from_numpy(context).float()
        y = self.nn_model(x).detach().numpy()
        n = context.shape[0]
        predicted_actions = np.argmax(y, axis=1)
        action_dist = np.zeros((n, self.n_actions, 1))
        action_dist[np.arange(n), predicted_actions, 0] = 1

        return action_dist

    def sample_action(
        self,
        context: np.ndarray,
        tau: Union[int, float] = 1.0,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Sample a ranking of (non-repetitive) actions from the Plackett-Luce ranking distribution.

        Note
        --------
        This `sample_action` method samples a **non-repetitive** ranking of actions for new data
        :math:`x \\in \\mathcal{X}` via the so-called "Gumbel Softmax trick" as follows.

        .. math::

            \\s (x,a) = \\hat{f}(x,a) / \\tau + \\gamma_{x,a}, \\quad \\gamma_{x,a} \\sim \\mathrm{Gumbel}(0,1)

        :math:`\\tau` is a temperature hyperparameter.
        :math:`f: \\mathcal{X} \\times \\mathcal{A} \\times \\mathcal{K} \\rightarrow \\mathbb{R}_{+}`
        is a scoring function which is now implemented in the `predict_score` method.
        When `len_list > 0`,  the expected rewards estimated at different positions will be averaged to form :math:`f(x,a)`.
        :math:`\\gamma_{x,a}` is a random variable sampled from the Gumbel distribution.
        By sorting the actions based on :math:`\\s (x,a)` for each context, we can efficiently sample a ranking from
        the Plackett-Luce ranking distribution.

        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        tau: int or float, default=1.0
            A temperature parameter that controls the randomness of the action choice
            by scaling the scores before applying softmax.
            As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

        random_state: int, default=None
            Controls the random seed in sampling actions.

        Returns
        -----------
        sampled_action: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Ranking of actions sampled from the Plackett-Luce ranking distribution via the Gumbel softmax trick.

        """
        check_array(array=context, name="context", expected_dim=2)
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )

        n = context.shape[0]
        random_ = check_random_state(random_state)
        sampled_action = np.zeros((n, self.n_actions, self.len_list))
        scores = self.predict_proba(context=context).mean(2) / tau
        scores += random_.gumbel(size=scores.shape)
        ranking = np.argsort(-scores, axis=1)
        for p in np.arange(self.len_list):
            sampled_action[np.arange(n), ranking[:, p], p] = 1
        return sampled_action

    def predict_proba(
        self,
        context: np.ndarray,
    ) -> np.ndarray:
        """Obtains action choice probabilities for new data.

        Note
        --------
        This policy uses multi-layer perceptron (MLP) and the softmax function as the last layer.
        This is a stochastic policy and represented as follows:

        .. math::

            \\pi_\\theta (a \\mid x) = \\frac{\\exp(f_\\theta(x, a))}{\\sum_{a' \\in \\mathcal{A}} \\exp(f_\\theta(x, a'))}

        where :math:`f__\\theta(x, a)` is MLP with parameter :math:`\\theta`.

        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        choice_prob: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choice probabilities obtained by a trained classifier.

        """
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )

        self.nn_model.eval()
        x = torch.from_numpy(context).float()
        y = self.nn_model(x).detach().numpy()
        return y[:, :, np.newaxis]


@dataclass
class QFuncEstimator:
    """Q-function estimator based on a neural network.

    Note
    --------
    The neural network is implemented in PyTorch.

    Parameters
    -----------
    n_actions: int
        Number of actions.

    dim_context: int
        Number of dimensions of context vectors.

    hidden_layer_size: Tuple[int, ...], default = (100,)
        The i-th element specifies the size of the i-th layer.

    activation: str, default='relu'
        Activation function.
        Must be one of the followings:
        - 'identity', the identity function, :math:`f(x) = x`.
        - 'logistic', the sigmoid function, :math:`f(x) = \\frac{1}{1 + \\exp(x)}`.
        - 'tanh', the hyperbolic tangent function, `:math:f(x) = \\frac{\\exp(x) - \\exp(-x)}{\\exp(x) + \\exp(-x)}`
        - 'relu', the rectified linear unit function, `:math:f(x) = \\max(0, x)`

    solver: str, default='adam'
        Optimizer of the neural network.
        Must be one of the followings:
        - 'sgd', Stochastic Gradient Descent.
        - 'adam', Adam (Kingma and Ba 2014).
        - 'adagrad', Adagrad (Duchi et al. 2011).

    alpha: float, default=0.001
        L2 penalty.

    batch_size: Union[int, str], default="auto"
        Batch size for SGD, Adagrad, and Adam.
        If "auto", the maximum of 200 and the number of samples is used.
        If integer, must be positive.

    learning_rate_init: int, default=0.0001
        Initial learning rate for SGD, Adagrad, and Adam.

    max_iter: int, default=200
        Number of epochs for SGD, Adagrad, and Adam.

    shuffle: bool, default=True
        Whether to shuffle samples in SGD and Adam.

    random_state: Optional[int], default=None
        Controls the random seed.

    tol: float, default=1e-4
        Tolerance for training.
        When the training loss is not improved at least `tol' for `n_iter_no_change' consecutive iterations,
        training is stopped.

    momentum: float, default=0.9
        Momentum for SGD.
        Must be in the range of [0., 1.].

    nesterovs_momentum: bool, default=True
        Whether to use Nesterov momentum.

    early_stopping: bool, default=False
        Whether to use early stopping for SGD, Adagrad, and Adam.
        If set to true, `validation_fraction' of training data is used as validation data,
        and training is stopped when the validation loss is not improved at least `tol' for `n_iter_no_change' consecutive iterations.

    validation_fraction: float, default=0.1
        Fraction of validation data when early stopping is used.
        Must be in the range of (0., 1.].

    beta_1: float, default=0.9
        Coefficient used for computing running average of gradient for Adam.
        Must be in the range of [0., 1.].

    beta_2: float, default=0.999
        Coefficient used for computing running average of the square of gradient for Adam.
        Must be in the range of [0., 1.].

    epsilon: float, default=1e-8
        Term for numerical stability in Adam.

    n_iter_no_change: int, default=10
        Maximum number of not improving epochs when early stopping is used.

    References
    ------------
    Dong .C. Liu and Jorge Nocedal.
    "On the Limited Memory Method for Large Scale Optimization.", 1989.

    Diederik P. Kingma and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization.", 2014.

    John Duchi, Elad Hazan, and Yoram Singer.
    "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization", 2011.

    """

    n_actions: int
    dim_context: int
    hidden_layer_size: Tuple[int, ...] = (100,)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: Union[int, str] = "auto"
    learning_rate_init: float = 0.0001
    max_iter: int = 200
    shuffle: bool = True
    random_state: Optional[int] = None
    tol: float = 1e-4
    momentum: float = 0.9
    nesterovs_momentum: bool = True
    early_stopping: bool = False
    validation_fraction: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    n_iter_no_change: int = 10

    def __post_init__(self) -> None:
        """Initialize class."""
        check_scalar(self.dim_context, "dim_context", int, min_val=1)

        if not isinstance(self.hidden_layer_size, tuple) or any(
            [not isinstance(h, int) or h <= 0 for h in self.hidden_layer_size]
        ):
            raise ValueError(
                f"`hidden_layer_size` must be a tuple of positive integers, but {self.hidden_layer_size} is given"
            )

        if self.solver not in ("adagrad", "sgd", "adam"):
            raise ValueError(
                f"`solver` must be one of 'adam', 'adagrad', or 'sgd', but {self.solver} is given"
            )

        check_scalar(self.alpha, "alpha", float, min_val=0.0)

        if self.batch_size != "auto" and (
            not isinstance(self.batch_size, int) or self.batch_size <= 0
        ):
            raise ValueError(
                f"`batch_size` must be a positive integer or 'auto', but {self.batch_size} is given"
            )

        check_scalar(self.learning_rate_init, "learning_rate_init", float)
        if self.learning_rate_init <= 0.0:
            raise ValueError(
                f"`learning_rate_init`= {self.learning_rate_init}, must be > 0.0"
            )

        check_scalar(self.max_iter, "max_iter", int, min_val=1)

        if not isinstance(self.shuffle, bool):
            raise ValueError(f"`shuffle` must be a bool, but {self.shuffle} is given")

        check_scalar(self.tol, "tol", float)
        if self.tol <= 0.0:
            raise ValueError(f"`tol`= {self.tol}, must be > 0.0")

        check_scalar(self.momentum, "momentum", float, min_val=0.0, max_val=1.0)

        if not isinstance(self.nesterovs_momentum, bool):
            raise ValueError(
                f"`nesterovs_momentum` must be a bool, but {self.nesterovs_momentum} is given"
            )

        if not isinstance(self.early_stopping, bool):
            raise ValueError(
                f"`early_stopping` must be a bool, but {self.early_stopping} is given"
            )

        check_scalar(
            self.validation_fraction, "validation_fraction", float, max_val=1.0
        )
        if self.validation_fraction <= 0.0:
            raise ValueError(
                f"`validation_fraction`= {self.validation_fraction}, must be > 0.0"
            )

        check_scalar(self.beta_1, "beta_1", float, min_val=0.0, max_val=1.0)
        check_scalar(self.beta_2, "beta_2", float, min_val=0.0, max_val=1.0)
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0)
        check_scalar(self.n_iter_no_change, "n_iter_no_change", int, min_val=1)

        if self.random_state is not None:
            self.random_ = check_random_state(self.random_state)
            torch.manual_seed(self.random_state)

        if self.activation == "identity":
            activation_layer = nn.Identity
        elif self.activation == "logistic":
            activation_layer = nn.Sigmoid
        elif self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU
        else:
            raise ValueError(
                "`activation` must be one of 'identity', 'logistic', 'tanh', 'relu', or 'elu'"
                f", but {self.activation} is given"
            )

        layer_list = []
        input_size = self.dim_context

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.n_actions)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

    def _create_train_data_for_q_func_estimation(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        **kwargs,
    ) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
        """Create training data for off-policy learning.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        Returns
        --------
        (training_data_loader, validation_data_loader): Tuple[DataLoader, Optional[DataLoader]]
            Training and validation data loaders in PyTorch

        """
        if self.batch_size == "auto":
            batch_size_ = min(200, context.shape[0])
        else:
            check_scalar(self.batch_size, "batch_size", int, min_val=1)
            batch_size_ = self.batch_size

        dataset = QFuncEstimatorDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(action).long(),
            torch.from_numpy(reward).float(),
        )

        if self.early_stopping:
            if context.shape[0] <= 1:
                raise ValueError(
                    f"the number of samples is too small ({context.shape[0]}) to create validation data"
                )

            validation_size = max(int(context.shape[0] * self.validation_fraction), 1)
            training_size = context.shape[0] - validation_size
            training_dataset, validation_dataset = torch.utils.data.random_split(
                dataset, [training_size, validation_size]
            )
            training_data_loader = torch.utils.data.DataLoader(
                training_dataset,
                batch_size=batch_size_,
                shuffle=self.shuffle,
            )
            validation_data_loader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=batch_size_,
                shuffle=self.shuffle,
            )

            return training_data_loader, validation_data_loader

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_,
            shuffle=self.shuffle,
        )

        return data_loader, None

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
    ) -> None:
        """Fits an offline bandit policy on the given logged bandit data.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
        )

        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )

        if self.solver == "sgd":
            optimizer = optim.SGD(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                momentum=self.momentum,
                weight_decay=self.alpha,
                nesterov=self.nesterovs_momentum,
            )
        elif self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                eps=self.epsilon,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.Adam(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                betas=(self.beta_1, self.beta_2),
                eps=self.epsilon,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError(
                "`solver` must be one of 'adam', 'adagrad', or 'sgd'"
            )

        (
            training_data_loader,
            validation_data_loader,
        ) = self._create_train_data_for_q_func_estimation(
            context,
            action,
            reward,
        )

        n_not_improving_training = 0
        previous_training_loss = None
        n_not_improving_validation = 0
        previous_validation_loss = None
        for _ in tqdm(np.arange(self.max_iter), desc="q-func learning"):
            self.nn_model.train()
            for x, a, r in training_data_loader:
                optimizer.zero_grad()
                q_hat = self.nn_model(x)[torch.arange(a.shape[0], dtype=torch.long), a]
                loss = mse_loss(r, q_hat)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                if previous_training_loss is not None:
                    if loss_value - previous_training_loss < self.tol:
                        n_not_improving_training += 1
                    else:
                        n_not_improving_training = 0
                if n_not_improving_training >= self.n_iter_no_change:
                    break
                previous_training_loss = loss_value

            if self.early_stopping:
                self.nn_model.eval()
                for x, a, r in validation_data_loader:
                    q_hat = self.nn_model(x)[
                        torch.arange(a.shape[0], dtype=torch.long), a
                    ]
                    loss = mse_loss(r, q_hat)
                    loss_value = loss.item()
                    if previous_validation_loss is not None:
                        if loss_value - previous_validation_loss < self.tol:
                            n_not_improving_validation += 1
                        else:
                            n_not_improving_validation = 0
                    if n_not_improving_validation > self.n_iter_no_change:
                        break
                    previous_validation_loss = loss_value

    def predict(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Predict best continuous actions for new data.

        Parameters
        -----------
        context: Tensor, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        estimated_expected_rewards: Tensor, shape (n_rounds_of_new_data,)
            Expected rewards given context and action for new data estimated by the regression model.

        """
        check_tensor(tensor=context, name="context", expected_dim=2)
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )

        self.nn_model.eval()
        return self.nn_model(context)


@dataclass
class NNPolicyDataset(torch.utils.data.Dataset):
    """PyTorch dataset for NNPolicyLearner"""

    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    position: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.position.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.position[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class QFuncEstimatorDataset(torch.utils.data.Dataset):
    """PyTorch dataset for QFuncEstimator"""

    feature: np.ndarray
    action: np.ndarray
    reward: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert self.feature.shape[0] == self.action.shape[0] == self.reward.shape[0]

    def __getitem__(self, index):
        return (
            self.feature[index],
            self.action[index],
            self.reward[index],
        )

    def __len__(self):
        return self.feature.shape[0]
