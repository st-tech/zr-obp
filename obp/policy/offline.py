# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Offline Bandit Algorithms."""
from collections import OrderedDict
from dataclasses import dataclass
import mypy_extensions as mx
from typing import Any, Callable, Tuple, Optional, Union

import numpy as np
from scipy.special import softmax
from sklearn.base import clone, ClassifierMixin, is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state, check_scalar
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .base import BaseOfflinePolicyLearner

from ..utils import check_bandit_feedback_inputs


@dataclass
class IPWLearner(BaseOfflinePolicyLearner):
    """Off-policy learner with Inverse Probability Weighting.

    Parameters
    -----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions recommended in each impression.
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
                raise ValueError("base_classifier must be a classifier")
        self.base_classifier_list = [
            clone(self.base_classifier) for _ in np.arange(self.len_list)
        ]

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
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Propensity scores, the probability of selecting each action by behavior policy,
            in the given logged bandit feedback.

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
        """Fits an offline bandit policy using the given logged bandit feedback data.

        Note
        --------
        This `fit` method trains a deterministic policy :math:`\\pi: \\mathcal{X} \\rightarrow \\mathcal{A}`
        via a cost-sensitive classification reduction as follows:

        .. math::

            \\hat{\\pi}
            & \\in \\arg \\max_{\\pi \\in \\Pi} \\hat{V}_{\\mathrm{IPW}} (\\pi ; \\mathcal{D}) \\\\
            & = \\arg \\max_{\\pi \\in \\Pi} \\mathbb{E}_{\\mathcal{D}} \\left[\\frac{\\mathbb{I} \\{\\pi (x_{i})=a_{i} \\}}{\\pi_{b}(a_{i} | x_{i})} r_{i} \\right] \\\\
            & = \\arg \\min_{\\pi \\in \\Pi} \\mathbb{E}_{\\mathcal{D}} \\left[\\frac{r_i}{\\pi_{b}(a_{i} | x_{i})} \\mathbb{I} \\{\\pi (x_{i}) \\neq a_{i} \\} \\right],

        where :math:`\\mathbb{E}_{\\mathcal{D}} [\cdot]` is the empirical average over observations in :math:`\\mathcal{D}`.
        See the reference for the details.


        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            If None is given, a learner assumes that there is only one position.
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
            n_actions = np.int(action.max() + 1)
            pscore = np.ones_like(action) / n_actions
        if position is None or self.len_list == 1:
            position = np.zeros_like(action, dtype=int)

        for position_ in np.arange(self.len_list):
            X, sample_weight, y = self._create_train_data_for_opl(
                context=context[position == position_],
                action=action[position == position_],
                reward=reward[position == position_],
                pscore=pscore[position == position_],
            )
            self.base_classifier_list[position_].fit(
                X=X, y=y, sample_weight=sample_weight
            )

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict best actions for new data.

        Note
        --------
        Action set predicted by this `predict` method can contain duplicate items.
        If you want a non-repetitive action set, then please use the `sample_action` method.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        action_dist: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choices by a classifier, which can contain duplicate items.
            If you want a non-repetitive action set, please use the `sample_action` method.

        """
        if not isinstance(context, np.ndarray) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional ndarray")

        n_rounds = context.shape[0]
        action_dist = np.zeros((n_rounds, self.n_actions, self.len_list))
        for position_ in np.arange(self.len_list):
            predicted_actions_at_position = self.base_classifier_list[
                position_
            ].predict(context)
            action_dist[
                np.arange(n_rounds),
                predicted_actions_at_position,
                np.ones(n_rounds, dtype=int) * position_,
            ] += 1
        return action_dist

    def predict_score(self, context: np.ndarray) -> np.ndarray:
        """Predict non-negative scores for all possible products of action and position.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        score_predicted: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Scores for all possible pairs of action and position predicted by a classifier.

        """
        assert (
            isinstance(context, np.ndarray) and context.ndim == 2
        ), "context must be 2-dimensional ndarray"

        n_rounds = context.shape[0]
        score_predicted = np.zeros((n_rounds, self.n_actions, self.len_list))
        for position_ in np.arange(self.len_list):
            score_predicteds_at_position = self.base_classifier_list[
                position_
            ].predict_proba(context)
            score_predicted[:, :, position_] = score_predicteds_at_position
        return score_predicted

    def sample_action(
        self,
        context: np.ndarray,
        tau: Union[int, float] = 1.0,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Sample (non-repetitive) actions based on scores predicted by a classifier.

        Note
        --------
        This `sample_action` method samples a **non-repetitive** set of actions for new data :math:`x \\in \\mathcal{X}`
        by first computing non-negative scores for all possible candidate products of action and position
        :math:`(a, k) \\in \\mathcal{A} \\times \\mathcal{K}` (where :math:`\\mathcal{A}` is an action set and
        :math:`\\mathcal{K}` is a position set), and using softmax function as follows:

        .. math::

            & P (A_1 = a_1 | x) = \\frac{\\mathrm{exp}(f(x,a_1,1) / \\tau)}{\\sum_{a^{\\prime} \\in \\mathcal{A}} \\mathrm{exp}( f(x,a^{\\prime},1) / \\tau)} , \\\\
            & P (A_2 = a_2 | A_1 = a_1, x) = \\frac{\\mathrm{exp}(f(x,a_2,2) / \\tau)}{\\sum_{a^{\\prime} \\in \\mathcal{A} \\backslash \\{a_1\\}} \\mathrm{exp}(f(x,a^{\\prime},2) / \\tau )} ,
            \\ldots

        where :math:`A_k` is a random variable representing an action at a position :math:`k`.
        :math:`\\tau` is a temperature hyperparameter.
        :math:`f: \\mathcal{X} \\times \\mathcal{A} \\times \\mathcal{K} \\rightarrow \\mathbb{R}_{+}`
        is a scoring function which is now implemented in the `predict_score` method.

        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        tau: int or float, default=1.0
            A temperature parameter, controlling the randomness of the action choice.
            As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

        random_state: int, default=None
            Controls the random seed in sampling actions.

        Returns
        -----------
        action: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action sampled by a trained classifier.

        """
        if not isinstance(context, np.ndarray) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional ndarray")
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        n_rounds = context.shape[0]
        random_ = check_random_state(random_state)
        action = np.zeros((n_rounds, self.n_actions, self.len_list))
        score_predicted = self.predict_score(context=context)
        for i in tqdm(np.arange(n_rounds), desc="[sample_action]", total=n_rounds):
            action_set = np.arange(self.n_actions)
            for position_ in np.arange(self.len_list):
                score_ = softmax(score_predicted[i, action_set, position_] / tau)
                action_sampled = random_.choice(action_set, p=score_, replace=False)
                action[i, action_sampled, position_] = 1
                action_set = np.delete(action_set, action_set == action_sampled)
        return action

    def predict_proba(
        self,
        context: np.ndarray,
        tau: Union[int, float] = 1.0,
    ) -> np.ndarray:
        """Obtains action choice probabilities for new data based on scores predicted by a classifier.

        Note
        --------
        This `predict_proba` method obtains action choice probabilities for new data :math:`x \\in \\mathcal{X}`
        by first computing non-negative scores for all possible candidate actions
        :math:`a \\in \\mathcal{A}` (where :math:`\\mathcal{A}` is an action set),
        and using a Plackett-Luce ranking model as follows:

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
            A temperature parameter, controlling the randomness of the action choice.
            As :math:`\\tau \\rightarrow \\infty`, the algorithm will select arms uniformly at random.

        Returns
        -----------
        choice_prob: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choice probabilities obtained by a trained classifier.

        """
        assert (
            self.len_list == 1
        ), "predict_proba method can be used only when len_list = 1"
        assert (
            isinstance(context, np.ndarray) and context.ndim == 2
        ), "context must be 2-dimensional ndarray"
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        score_predicted = self.predict_score(context=context)
        choice_prob = softmax(score_predicted / tau, axis=1)
        return choice_prob


@dataclass
class NNPolicyLearner(BaseOfflinePolicyLearner):
    """Off-policy learner using a neural network whose objective function is an OPE estimator.

    Note
    --------
    The neural network is implemented in PyTorch.

    Parameters
    -----------
    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions recommended in each impression.
        When Open Bandit Dataset is used, 3 should be set.
        Currently, len_list > 1 is not supported.

    dim_context: int
        Number of dimensions of context vectors.

    off_policy_objective: Callable[[VarArg[Any]], Tensor]
        Function returns the value of an OPE estimator.
        `BaseOffPolicyEstimator.estimate_policy_value_tensor` is supposed to be given here.

    hidden_layer_size: Tuple[int, ...], default = (100,)
        The i th element specifies the size of the i th layer.

    activation: str, default='relu'
        Activation function.
        Must be one of the following:

        - 'identity', the identity function, :math:`f(x) = x`.
        - 'logistic', the sigmoid function, :math:`f(x) = \\frac{1}{1 + \\exp(x)}`.
        - 'tanh', the hyperbolic tangent function, `:math:f(x) = \\frac{\\exp(x) - \\exp(-x)}{\\exp(x) + \\exp(-x)}`
        - 'relu', the rectified linear unit function, `:math:f(x) = \\max(0, x)`

    solver: str, default='adam'
        Optimizer of the neural network.
        Must be one of the following:

        - 'lbfgs', L-BFGS algorithm (Liu and Nocedal 1989).
        - 'sgd', stochastic gradient descent (SGD).
        - 'adam', Adam (Kingma and Ba 2014).

    alpha: float, default=0.001
        L2 penalty.

    batch_size: Union[int, str], default="auto"
        Batch size for SGD and Adam.
        If "auto", the maximum of 200 and the number of samples is used.
        If integer, must be positive.

    learning_rate_init: int, default=0.0001
        Initial learning rate for SGD and Adam.

    max_iter: int, default=200
        Maximum number of iterations for L-BFGS.
        Number of epochs for SGD and Adam.

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
        Whether to use early stopping for SGD and Adam.
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

    max_fun: int, default=15000
        Maximum number of function calls per step in L-BFGS.

    References:
    ------------
    Dong .C. Liu and Jorge Nocedal.
    "On the Limited Memory Method for Large Scale Optimization.", 1989

    Diederik P. Kingma and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization.", 2014
    """

    dim_context: Optional[int] = None
    off_policy_objective: Optional[Callable[[mx.VarArg(Any)], torch.Tensor]] = None
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
    max_fun: int = 15000

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

        if self.len_list != 1:
            raise NotImplementedError("currently, len_list > 1 is not supported")

        if not isinstance(self.dim_context, int) or self.dim_context <= 0:
            raise ValueError(
                f"dim_context must be a positive integer, but {self.dim_context} is given"
            )

        if not callable(self.off_policy_objective):
            raise ValueError(
                f"off_policy_objective must be callable, but {self.off_policy_objective} is given"
            )

        if not isinstance(self.hidden_layer_size, tuple) or any(
            [not isinstance(h, int) or h <= 0 for h in self.hidden_layer_size]
        ):
            raise ValueError(
                f"hidden_layer_size must be tuple of positive integers, but {self.hidden_layer_size} is given"
            )

        if self.solver not in ("lbfgs", "sgd", "adam"):
            raise ValueError(
                f"solver must be one of 'adam', 'lbfgs', or 'sgd', but {self.solver} is given"
            )

        if not isinstance(self.alpha, float) or self.alpha < 0.0:
            raise ValueError(
                f"alpha must be a non-negative float, but {self.alpha} is given"
            )

        if self.batch_size != "auto" and (
            not isinstance(self.batch_size, int) or self.batch_size <= 0
        ):
            raise ValueError(
                f"batch_size must be a positive integer or 'auto', but {self.batch_size} is given"
            )

        if (
            not isinstance(self.learning_rate_init, float)
            or self.learning_rate_init <= 0.0
        ):
            raise ValueError(
                f"learning_rate_init must be a positive float, but {self.learning_rate_init} is given"
            )

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError(
                f"max_iter must be a positive integer, but {self.max_iter} is given"
            )

        if not isinstance(self.shuffle, bool):
            raise ValueError(f"shuffle must be a bool, but {self.shuffle} is given")

        if not isinstance(self.tol, float) or self.tol <= 0.0:
            raise ValueError(f"tol must be a positive float, but {self.tol} is given")

        if not isinstance(self.momentum, float) or not 0.0 <= self.momentum <= 1.0:
            raise ValueError(
                f"momentum must be a float in [0., 1.], but {self.momentum} is given"
            )

        if not isinstance(self.nesterovs_momentum, bool):
            raise ValueError(
                f"nesterovs_momentum must be a bool, but {self.nesterovs_momentum} is given"
            )

        if not isinstance(self.early_stopping, bool):
            raise ValueError(
                f"early_stopping must be a bool, but {self.early_stopping} is given"
            )

        if self.early_stopping and self.solver not in ("sgd", "adam"):
            raise ValueError(
                f"if early_stopping is True, solver must be one of 'sgd' or 'adam', but {self.solver} is given"
            )

        if (
            not isinstance(self.validation_fraction, float)
            or not 0.0 < self.validation_fraction <= 1.0
        ):
            raise ValueError(
                f"validation_fraction must be a float in (0., 1.], but {self.validation_fraction} is given"
            )

        if not isinstance(self.beta_1, float) or not 0.0 <= self.beta_1 <= 1.0:
            raise ValueError(
                f"beta_1 must be a float in [0. 1.], but {self.beta_1} is given"
            )

        if not isinstance(self.beta_2, float) or not 0.0 <= self.beta_2 <= 1.0:
            raise ValueError(
                f"beta_2 must be a float in [0., 1.], but {self.beta_2} is given"
            )

        if not isinstance(self.beta_2, float) or not 0.0 <= self.beta_2 <= 1.0:
            raise ValueError(
                f"beta_2 must be a float in [0., 1.], but {self.beta_2} is given"
            )

        if not isinstance(self.epsilon, float) or self.epsilon < 0.0:
            raise ValueError(
                f"epsilon must be a non-negative float, but {self.epsilon} is given"
            )

        if not isinstance(self.n_iter_no_change, int) or self.n_iter_no_change <= 0:
            raise ValueError(
                f"n_iter_no_change must be a positive integer, but {self.n_iter_no_change} is given"
            )

        if not isinstance(self.max_fun, int) or self.max_fun <= 0:
            raise ValueError(
                f"max_fun must be a positive integer, but {self.max_fun} is given"
            )

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
        else:
            raise ValueError(
                f"activation must be one of 'identity', 'logistic', 'tanh', or 'relu', but {self.activation} is given"
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

    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        position: np.ndarray,
        **kwargs,
    ) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
        """Create training data for off-policy learning.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Propensity scores, the probability of selecting each action by behavior policy,
            in the given logged bandit feedback.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            If None is given, a learner assumes that there is only one position.

        Returns
        --------
        (training_data_loader, validation_data_loader): Tuple[DataLoader, Optional[DataLoader]]
            Training and validation data loaders in PyTorch
        """
        if self.batch_size == "auto":
            batch_size_ = min(200, context.shape[0])
        elif isinstance(self.batch_size, int) and self.batch_size > 0:
            batch_size_ = self.batch_size
        else:
            raise ValueError("batch_size must be a positive integer or 'auto'")

        dataset = NNPolicyDataset(
            torch.from_numpy(context).float(),
            action,
            torch.from_numpy(reward).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(estimated_rewards_by_reg_model).float(),
            position,
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
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
    ) -> None:
        """Fits an offline bandit policy using the given logged bandit feedback data.

        Note
        ----------
        Given the training data :math:`\\mathcal{D}`, this policy maximizes the following objective function:

        .. math::

            \\hat{V}(\\pi_\\theta; \\mathcal{D}) - \\alpha \\Omega(\\theta)

        where :math:`\\hat{V}` is an OPE estimator and :math:`\\alpha \\Omega(\\theta)` is a regularization term.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities of behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards given context, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
            If None is given, a learner assumes that the estimated rewards are zero.

        position: array-like, shape (n_rounds,), default=None
            Position of recommendation interface where action was presented in each round of the given logged bandit feedback.
            If None is given, a learner assumes that there is only one position.
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
                "the second dimension of context must be equal to dim_context"
            )

        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions
        if estimated_rewards_by_reg_model is None:
            estimated_rewards_by_reg_model = np.zeros(
                (context.shape[0], self.n_actions, self.len_list)
            )

        if self.len_list == 1:
            position = np.zeros_like(action, dtype=int)
        else:
            raise NotImplementedError("currently, len_list > 1 is not supported")

        if self.solver == "lbfgs":
            optimizer = optim.LBFGS(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                max_iter=self.max_iter,
                max_eval=self.max_fun,
            )
        elif self.solver == "sgd":
            optimizer = optim.SGD(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                momentum=self.momentum,
                weight_decay=self.alpha,
                nesterov=self.nesterovs_momentum,
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
            raise NotImplementedError("solver must be one of 'adam', 'lbfgs', or 'sgd'")

        training_data_loader, validation_data_loader = self._create_train_data_for_opl(
            context, action, reward, pscore, estimated_rewards_by_reg_model, position
        )

        if self.solver == "lbfgs":
            for x, a, r, p, q_hat, pos in training_data_loader:

                def closure():
                    optimizer.zero_grad()
                    action_dist = self.nn_model(x).unsqueeze(-1)
                    loss = -1.0 * self.off_policy_objective(
                        reward=r,
                        action=a,
                        pscore=p,
                        action_dist=action_dist,
                        estimated_rewards_by_reg_model=q_hat,
                        position=pos,
                    )
                    loss.backward()
                    return loss

                optimizer.step(closure)
        if self.solver in ("sgd", "adam"):
            n_not_improving_training = 0
            previous_training_loss = None
            n_not_improving_validation = 0
            previous_validation_loss = None
            for _ in np.arange(self.max_iter):
                self.nn_model.train()
                for x, a, r, p, q_hat, pos in training_data_loader:
                    optimizer.zero_grad()
                    action_dist = self.nn_model(x).unsqueeze(-1)
                    loss = -1.0 * self.off_policy_objective(
                        reward=r,
                        action=a,
                        pscore=p,
                        action_dist=action_dist,
                        estimated_rewards_by_reg_model=q_hat,
                        position=pos,
                    )
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
                    for x, a, r, p, q_hat, pos in validation_data_loader:
                        action_dist = self.nn_model(x).unsqueeze(-1)
                        loss = -1.0 * self.off_policy_objective(
                            reward=r,
                            action=a,
                            pscore=p,
                            action_dist=action_dist,
                            estimated_rewards_by_reg_model=q_hat,
                            position=pos,
                        )
                        loss_value = loss.item()
                        if previous_validation_loss is not None:
                            if loss_value - previous_validation_loss < self.tol:
                                n_not_improving_validation += 1
                            else:
                                n_not_improving_validation = 0
                        if n_not_improving_validation > self.n_iter_no_change:
                            break
                        previous_validation_loss = loss_value

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict best actions for new data.

        Note
        --------
        Action set predicted by this `predict` method can contain duplicate items.
        If you want a non-repetitive action set, then please use the `sample_action` method.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        action_dist: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action choices by a classifier, which can contain duplicate items.
            If you want a non-repetitive action set, please use the `sample_action` method.

        """
        if not isinstance(context, np.ndarray) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional ndarray")

        if context.shape[1] != self.dim_context:
            raise ValueError(
                "the second dimension of context must be equal to dim_context"
            )

        self.nn_model.eval()
        x = torch.from_numpy(context).float()
        y = self.nn_model(x).detach().numpy()
        predicted_actions = np.argmax(y, axis=1)
        n_rounds = context.shape[0]
        action_dist = np.zeros((n_rounds, self.n_actions, 1))
        action_dist[np.arange(n_rounds), predicted_actions, 0] = 1

        return action_dist

    def sample_action(
        self,
        context: np.ndarray,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Sample (non-repetitive) actions based on action choice probabilities.

        Parameters
        ----------------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        random_state: int, default=None
            Controls the random seed in sampling actions.

        Returns
        -----------
        action: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Action sampled by a trained classifier.

        """
        if not isinstance(context, np.ndarray) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional ndarray")

        if context.shape[1] != self.dim_context:
            raise ValueError(
                "the second dimension of context must be equal to dim_context"
            )

        n_rounds = context.shape[0]
        random_ = check_random_state(random_state)
        action = np.zeros((n_rounds, self.n_actions, self.len_list))
        score_predicted = self.predict_proba(context=context)
        for i in tqdm(np.arange(n_rounds), desc="[sample_action]", total=n_rounds):
            action_set = np.arange(self.n_actions)
            for position_ in np.arange(self.len_list):
                score_ = score_predicted[i, action_set, position_]
                action_sampled = random_.choice(action_set, p=score_, replace=False)
                action[i, action_sampled, position_] = 1
                action_set = np.delete(action_set, action_set == action_sampled)
        return action

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
        if not isinstance(context, np.ndarray) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional ndarray")

        if context.shape[1] != self.dim_context:
            raise ValueError(
                "the second dimension of context must be equal to dim_context"
            )

        self.nn_model.eval()
        x = torch.from_numpy(context).float()
        y = self.nn_model(x).detach().numpy()
        return y[:, :, np.newaxis]


@dataclass
class NNPolicyDataset(torch.utils.data.Dataset):
    """PyTorch dataset for NNPolicyLearner"""

    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    estimated_rewards_by_reg_model: np.ndarray
    position: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.estimated_rewards_by_reg_model.shape[0]
            == self.position.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.estimated_rewards_by_reg_model[index],
            self.position[index],
        )

    def __len__(self):
        return self.context.shape[0]
