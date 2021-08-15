# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Offline(Batch) Policy Learning for Continuous Action."""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Dict
from tqdm import tqdm

import numpy as np
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseContinuousOfflinePolicyLearner
from ..utils import check_continuous_bandit_feedback_inputs


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
        Whether to use Nesterov momentum.

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

    q_func_estimator_hyperparams: Dict, default=None
        A set of hyperparameters to define q function estimator.

    References:
    ------------
    Dong .C. Liu and Jorge Nocedal.
    "On the Limited Memory Method for Large Scale Optimization.", 1989.

    Diederik P. Kingma and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization.", 2014.

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
    q_func_estimator_hyperparams: Optional[Dict] = None

    def __post_init__(self) -> None:
        """Initialize class."""
        if not isinstance(self.dim_context, int) or self.dim_context <= 0:
            raise ValueError(
                f"dim_context must be a positive integer, but {self.dim_context} is given"
            )

        if self.pg_method not in ["dpg", "ipw", "dr"]:
            raise ValueError(
                f"pg_method must be one of 'dgp', 'ipw', or 'dr', but {self.pg_method} is given"
            )

        if self.pg_method != "dpg":
            if not isinstance(self.bandwidth, (int, float)) or self.bandwidth <= 0:
                raise ValueError(
                    f"bandwidth must be a positive float, but {self.bandwidth} is given"
                )

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

        if self.q_func_estimator_hyperparams is not None:
            if not isinstance(self.q_func_estimator_hyperparams, dict):
                raise ValueError(
                    f"q_func_estimator_hyperparams must be a dict, but {type(self.q_func_estimator_hyperparams)} is given"
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
                f"activation must be one of 'identity', 'logistic', 'tanh', 'relu', or 'elu' but {self.activation} is given"
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
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Propensity scores, the probability of selecting each action by behavior policy in the given logged bandit feedback.

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

        action: array-like or Tensor, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities by a behavior policy (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        """
        check_continuous_bandit_feedback_inputs(
            context=context,
            action_by_behavior_policy=action,
            reward=reward,
            pscore=pscore,
        )

        if context.shape[1] != self.dim_context:
            raise ValueError(
                "the second dimension of context must be equal to dim_context"
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
            context,
            action,
            reward,
            pscore,
        )

        if self.solver == "lbfgs":
            for x, a, r, p in training_data_loader:

                def closure():
                    optimizer.zero_grad()
                    action_by_current_policy = self.nn_model(x).flatten()
                    loss = -1.0 * self._estimate_policy_value(
                        context=x,
                        reward=r,
                        action=a,
                        pscore=p,
                        action_by_current_policy=action_by_current_policy,
                    )
                    loss.backward()
                    return loss

                optimizer.step(closure)
        if self.solver in ("sgd", "adam"):
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
                    loss = -1.0 * self._estimate_policy_value(
                        context=x,
                        reward=r,
                        action=a,
                        pscore=p,
                        action_by_current_policy=action_by_current_policy,
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
                    for x, a, r, p in validation_data_loader:
                        action_by_current_policy = self.nn_model(x).flatten()
                        loss = -1.0 * self._estimate_policy_value(
                            context=x,
                            reward=r,
                            action=a,
                            pscore=p,
                            action_by_current_policy=action_by_current_policy,
                        )
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

    def _estimate_policy_value(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        pscore: torch.Tensor,
        action_by_current_policy: torch.Tensor,
    ) -> float:
        """Calculate policy loss used in the policy gradient method.

        Parameters
        -----------
        context: Tensor, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: Tensor, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: Tensor, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: Tensor, shape (n_rounds,)
            Action choice probabilities of a behavior policy (generalized propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        action_by_current_policy: array-like or Tensor, shape (n_rounds,)
            Continuous action values given by the current policy.

        """

        def gaussian_kernel(u: torch.Tensor) -> torch.Tensor:
            return torch.exp(-(u ** 2) / 2) / ((2 * np.pi) ** 0.5)

        if self.output_space is not None:
            action_by_current_policy = torch.clamp(
                action_by_current_policy,
                min=self.output_space[0],
                max=self.output_space[1],
            )

        if self.pg_method == "dpg":
            estimated_policy_value = self.q_func_estimator.predict(
                context=context,
                action=action_by_current_policy,
            )

        elif self.pg_method == "ipw":
            u = action_by_current_policy - action
            u /= self.bandwidth
            estimated_policy_value = gaussian_kernel(u) * reward / pscore
            estimated_policy_value /= self.bandwidth

        elif self.pg_method == "dr":
            u = action_by_current_policy - action
            u /= self.bandwidth
            q_hat = self.q_func_estimator.predict(
                context=context,
                action=action_by_current_policy,
            )
            estimated_policy_value = gaussian_kernel(u) * (reward - q_hat) / pscore
            estimated_policy_value /= self.bandwidth
            estimated_policy_value += q_hat

        return estimated_policy_value.mean()

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
        if not isinstance(context, np.ndarray) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional ndarray")

        if context.shape[1] != self.dim_context:
            raise ValueError(
                "the second dimension of context must be equal to dim_context"
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
        Whether to use Nesterov momentum.

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

    References
    ------------
    Dong .C. Liu and Jorge Nocedal.
    "On the Limited Memory Method for Large Scale Optimization.", 1989.

    Diederik P. Kingma and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization.", 2014.

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
        if not isinstance(self.dim_context, int) or self.dim_context <= 0:
            raise ValueError(
                f"dim_context must be a positive integer, but {self.dim_context} is given"
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
        elif self.activation == "elu":
            activation_layer = nn.ELU
        else:
            raise ValueError(
                f"activation must be one of 'identity', 'logistic', 'tanh', 'relu', or 'elu', but {self.activation} is given"
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
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

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
        """Fits an offline bandit policy using the given logged bandit feedback data.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like or Tensor, shape (n_rounds,)
            Continuous action values sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        """
        check_continuous_bandit_feedback_inputs(
            context=context,
            action_by_behavior_policy=action,
            reward=reward,
        )

        if context.shape[1] != self.dim_context:
            raise ValueError(
                "the second dimension of context must be equal to dim_context"
            )

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

        (
            training_data_loader,
            validation_data_loader,
        ) = self._create_train_data_for_q_func_estimation(
            context,
            action,
            reward,
        )

        if self.solver == "lbfgs":
            for x, r in training_data_loader:

                def closure():
                    optimizer.zero_grad()
                    q_hat = self.nn_model(x).flatten()
                    loss = nn.functional.mse_loss(r, q_hat)
                    loss.backward()
                    return loss

                optimizer.step(closure)
        if self.solver in ("sgd", "adam"):
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
        if not isinstance(context, torch.Tensor) or context.ndim != 2:
            raise ValueError("context must be 2-dimensional Tensor")

        if context.shape[1] != self.dim_context:
            raise ValueError(
                "the second dimension of context must be equal to dim_context"
            )

        if not isinstance(action, torch.Tensor) or action.ndim != 1:
            raise ValueError("action must be 1-dimensional Tensor")

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
