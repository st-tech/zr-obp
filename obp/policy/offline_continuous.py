# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Offline(Batch) Policy Learning for Continuous Action."""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..utils import check_array
from ..utils import check_continuous_bandit_feedback_inputs
from ..utils import check_tensor
from .base import BaseContinuousOfflinePolicyLearner


@dataclass
class ContinuousNNPolicyLearner(BaseContinuousOfflinePolicyLearner):
    """Off-policy learner using a neural network whose objective function is an OPE estimator.

    Note
    --------
    The neural network is implemented in PyTorch.

    Parameters
    -----------
    dim_context: int
        Number of dimensions of context vectors.

    pg_method: str
        A policy gradient method to train a neural network policy.
        Must be one of "dpg", "ipw", or "dr".
        See Kallus and Uehara.(2020) for the detailed description of these methods.
        "dpg" stands for Deterministic Policy Gradient.
        "ipw" corresponds to Importance Sampling Policy Gradient (ISPG) of Kallus and Uehara.(2020).
        "dr" corresponds to Eq.(7) of Kallus and Uehara.(2020).

    bandwidth: float, default=None
        A bandwidth hyperparameter used to kernelize the deterministic policy.
        A larger value increases bias instead of reducing variance.
        A smaller value increases variance instead of reducing bias.
        When pg_method is either "ipw" or "dr", a float value must be given.

    output_space: Tuple[Union[int, float], Uniton[int, float]], default=None
        Output space of the neural network policy.

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

    max_iter: int, default=100
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

    q_func_estimator_hyperparams: Dict, default=None
        A set of hyperparameters to define q function estimator.

    References:
    ------------
    Dong .C. Liu and Jorge Nocedal.
    "On the Limited Memory Method for Large Scale Optimization.", 1989.

    Diederik P. Kingma and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization.", 2014.

    John Duchi, Elad Hazan, and Yoram Singer.
    "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization", 2011.

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments", 2018.

    Nathan Kallus and Masatoshi Uehara.
    "Doubly Robust Off-Policy Value and Gradient Estimation for Deterministic Policies", 2020.

    """

    dim_context: int
    pg_method: str
    bandwidth: Optional[float] = None
    output_space: Tuple[Union[int, float], Union[int, float]] = None
    hidden_layer_size: Tuple[int, ...] = (100,)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: Union[int, str] = "auto"
    learning_rate_init: float = 0.0001
    max_iter: int = 100
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
        check_scalar(self.dim_context, "dim_context", int, min_val=1)

        if self.pg_method not in ["dpg", "ipw", "dr"]:
            raise ValueError(
                f"pg_method must be one of 'dgp', 'ipw', or 'dr', but {self.pg_method} is given"
            )

        if self.pg_method != "dpg":
            check_scalar(self.bandwidth, "bandwidth", (int, float))
            if self.bandwidth <= 0:
                raise ValueError(f"`bandwidth`= {self.bandwidth}, must be > 0.")

        if self.output_space is not None:
            if not isinstance(self.output_space, tuple) or any(
                [not isinstance(o, (int, float)) for o in self.output_space]
            ):
                raise ValueError(
                    f"output_space must be tuple of integers or floats, but {self.output_space} is given"
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

        check_scalar(self.beta_1, "beta_1", float, min_val=0.0, max_val=1.0)
        check_scalar(self.beta_2, "beta_2", float, min_val=0.0, max_val=1.0)
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0)
        check_scalar(self.n_iter_no_change, "n_iter_no_change", int, min_val=1)

        if self.q_func_estimator_hyperparams is not None:
            if not isinstance(self.q_func_estimator_hyperparams, dict):
                raise ValueError(
                    "`q_func_estimator_hyperparams` must be a dict"
                    f", but {type(self.q_func_estimator_hyperparams)} is given"
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
        elif self.activation == "elu":
            activation_layer = nn.ELU
        else:
            raise ValueError(
                "`activation` must be one of 'identity', 'logistic', 'tanh', 'relu', or 'elu'"
            )

        layer_list = []
        input_size = self.dim_context

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, 1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        if self.pg_method != "ipw":
            if self.q_func_estimator_hyperparams is not None:
                self.q_func_estimator_hyperparams["dim_context"] = self.dim_context
                self.q_func_estimator = QFuncEstimatorForContinuousAction(
                    **self.q_func_estimator_hyperparams
                )
            else:
                self.q_func_estimator = QFuncEstimatorForContinuousAction(
                    dim_context=self.dim_context
                )

    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
    ) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
        """Create training data for off-policy learning.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Continuous action values sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Propensity scores, the probability of selecting each action by behavior policy in the given logged bandit data.

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
            raise ValueError("`batch_size` must be a positive integer or 'auto'")

        dataset = NNPolicyDatasetForContinuousAction(
            torch.from_numpy(context).float(),
            torch.from_numpy(action).float(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(pscore).float(),
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
            Continuous action values sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities by a behavior policy (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        """
        check_continuous_bandit_feedback_inputs(
            context=context,
            action_by_behavior_policy=action,
            reward=reward,
            pscore=pscore,
        )

        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )

        if pscore is None:
            pscore = np.ones_like(action)

        # train q function estimator when it is needed to train NNPolicy
        if self.pg_method != "ipw":
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
            context,
            action,
            reward,
            pscore,
        )

        n_not_improving_training = 0
        previous_training_loss = None
        n_not_improving_validation = 0
        previous_validation_loss = None
        self.val_loss_curve = list()
        for _ in tqdm(np.arange(self.max_iter), desc="policy learning"):
            self.nn_model.train()
            for x, a, r, p in training_data_loader:
                optimizer.zero_grad()
                action_by_current_policy = self.nn_model(x).flatten()
                loss = -self._estimate_policy_gradient(
                    context=x,
                    reward=r,
                    action=a,
                    pscore=p,
                    action_by_current_policy=action_by_current_policy,
                ).mean()
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
                for x, a, r, p in validation_data_loader:
                    action_by_current_policy = self.nn_model(x).flatten()
                    loss = -self._estimate_policy_gradient(
                        context=x,
                        reward=r,
                        action=a,
                        pscore=p,
                        action_by_current_policy=action_by_current_policy,
                    ).mean()
                    loss_value = loss.item()
                    self.val_loss_curve.append(-loss_value)
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
        action_by_current_policy: torch.Tensor,
    ) -> float:
        """Estimate the policy gradient.

        Parameters
        -----------
        context: Tensor, shape (batch_size, dim_context)
            Context vectors observed for each data, i.e., :math:`x_i`.

        action: Tensor, shape (batch_size,)
            Continuous action values sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: Tensor, shape (batch_size,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: Tensor, shape (batch_size,)
            Action choice probabilities of the logging/behavior policy (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_by_current_policy: Tensor, shape (batch_size,)
            Continuous action values given by the current policy.

        Returns
        ----------
        estimated_policy_grad_arr: array-like, shape (batch_size,)
            Rewards of each data estimated by an OPE estimator.

        """

        def gaussian_kernel(u: torch.Tensor) -> torch.Tensor:
            return torch.exp(-(u**2) / 2) / ((2 * np.pi) ** 0.5)

        if self.output_space is not None:
            action_by_current_policy = torch.clamp(
                action_by_current_policy,
                min=self.output_space[0],
                max=self.output_space[1],
            )

        if self.pg_method == "dpg":
            estimated_policy_grad_arr = self.q_func_estimator.predict(
                context=context,
                action=action_by_current_policy,
            )

        elif self.pg_method == "ipw":
            u = action_by_current_policy - action
            u /= self.bandwidth
            estimated_policy_grad_arr = gaussian_kernel(u) * reward / pscore
            estimated_policy_grad_arr /= self.bandwidth

        elif self.pg_method == "dr":
            u = action_by_current_policy - action
            u /= self.bandwidth
            q_hat = self.q_func_estimator.predict(
                context=context,
                action=action_by_current_policy,
            )
            estimated_policy_grad_arr = gaussian_kernel(u) * (reward - q_hat) / pscore
            estimated_policy_grad_arr /= self.bandwidth
            estimated_policy_grad_arr += q_hat

        return estimated_policy_grad_arr

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict best continuous actions for new data.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        Returns
        -----------
        predicted_actions: array-like, shape (n_rounds_of_new_data,)
            Continuous action values given by a neural network policy.

        """
        check_array(array=context, name="context", expected_dim=2)
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )

        self.nn_model.eval()
        x = torch.from_numpy(context).float()
        predicted_actions = self.nn_model(x).detach().numpy().flatten()
        if self.output_space is not None:
            predicted_actions = np.clip(
                predicted_actions,
                a_min=self.output_space[0],
                a_max=self.output_space[1],
            )

        return predicted_actions


@dataclass
class QFuncEstimatorForContinuousAction:
    """Q-function estimator using a neural network for continuous action settings.

    Note
    --------
    The neural network is implemented in PyTorch.

    Parameters
    -----------
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

    max_iter: int, default=100
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

    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments", 2018.

    Nathan Kallus and Masatoshi Uehara.
    "Doubly Robust Off-Policy Value and Gradient Estimation for Deterministic Policies", 2020.

    """

    dim_context: int
    hidden_layer_size: Tuple[int, ...] = (100,)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 0.0001
    batch_size: Union[int, str] = "auto"
    learning_rate_init: float = 0.0001
    max_iter: int = 100
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
        input_size = self.dim_context + 1

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, 1)))

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
            Continuous action values sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

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
            raise ValueError("`batch_size` must be a positive integer or 'auto'")

        feature = np.c_[context, action[:, np.newaxis]]
        dataset = QFuncEstimatorDatasetForContinuousAction(
            torch.from_numpy(feature).float(),
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
            Continuous action values sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        """
        check_continuous_bandit_feedback_inputs(
            context=context,
            action_by_behavior_policy=action,
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
            for x, r in training_data_loader:
                optimizer.zero_grad()
                q_hat = self.nn_model(x).flatten()
                loss = nn.functional.mse_loss(r, q_hat)
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
                for x, r in validation_data_loader:
                    q_hat = self.nn_model(x).flatten()
                    loss = nn.functional.mse_loss(r, q_hat)
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
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict best continuous actions for new data.

        Parameters
        -----------
        context: Tensor, shape (n_rounds_of_new_data, dim_context)
            Context vectors for new data.

        action: Tensor, shape (n_rounds,)
            Continuous action values for new data.

        Returns
        -----------
        predicted_rewards: Tensor, shape (n_rounds_of_new_data,)
            Expected rewards given context and action for new data estimated by the regression model.

        """
        check_tensor(tensor=context, name="context", expected_dim=2)
        check_tensor(tensor=action, name="action", expected_dim=1)
        if context.shape[1] != self.dim_context:
            raise ValueError(
                "Expected `context.shape[1] == self.dim_context`, but found it False"
            )

        self.nn_model.eval()
        x = torch.cat((context, action.unsqueeze(-1)), 1)
        predicted_rewards = self.nn_model(x).flatten()

        return predicted_rewards


@dataclass
class NNPolicyDatasetForContinuousAction(torch.utils.data.Dataset):
    """PyTorch dataset for NNPolicyLearnerForContinuousAction"""

    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class QFuncEstimatorDatasetForContinuousAction(torch.utils.data.Dataset):
    """PyTorch dataset for QFuncEstimatorForContinuousAction"""

    feature: np.ndarray
    reward: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert self.feature.shape[0] == self.reward.shape[0]

    def __getitem__(self, index):
        return (
            self.feature[index],
            self.reward[index],
        )

    def __len__(self):
        return self.feature.shape[0]
