# Licensed under the Apache 2.0 License.

"""Offline Bandit Algorithms."""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Optional, Union

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

    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create training data for off-policy learning.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

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
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.
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
        if self.len_list == 1:
            position = np.zeros_like(action, dtype=int)
        else:
            if not isinstance(position, np.ndarray) or position.ndim != 1:
                raise ValueError(
                    f"when len_list > 1, position must be a 1-dimensional ndarray"
                )

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
        ), f"predict_proba method can be used only when len_list = 1"
        assert (
            isinstance(context, np.ndarray) and context.ndim == 2
        ), "context must be 2-dimensional ndarray"
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        score_predicted = self.predict_score(context=context)
        choice_prob = softmax(score_predicted / tau, axis=1)
        return choice_prob


@dataclass
class NNPolicyLearner(BaseOfflinePolicyLearner):
    """Off-policy learner using neural networks with off-policy estimators."""

    context_size: int
    objective = doubly_robust_tensor
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
    nestrovs_momentum: bool = True
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

        activation_layer = None

        if self.activation == "identity":
            activation_layer = nn.Identity
        elif self.activation == "logistic":
            activation_layer = nn.Sigmoid
        elif self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        else:
            raise NotImplementedError(
                "activation should be one of 'identity', 'logistic', 'tanh', or 'relu'"
            )

        layer_list = []
        input_size = self.context_size
        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
        layer_list.append(("output", nn.Softmax()))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        if self.solver == "adam":
            self.optimizer = optim.Adam(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                betas=(self.beta_1, beta_2),
                eps=self.epsilon,
                weight_decay=self.alpha,
            )
        elif self.solver == "lbfgs":
            self.optimizer = optim.LBFGS(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                max_iter=self.max_iter,
                max_eval=self.max_fun,
            )
        elif self.solver == "sgd":
            self.optimizer = optim.SGD(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                momentum=self.momentum,
                weight_decay=self.alpha,
                nesterov=self.nestrovs_momentum,
            )
        else:
            raise NotImplementedError(
                "solver should be one of 'adam', 'lbfgs', or 'sgd'"
            )

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
    ) -> None:
        """Fits an offline bandit policy using the given logged bandit feedback data.

        Parameters
        -----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors in each round, i.e., :math:`x_t`.

        action: array-like, shape (n_rounds,)
            Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

        reward: array-like, shape (n_rounds,)
            Observed rewards (or outcome) in each round, i.e., :math:`r_t`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

        position: array-like, shape (n_rounds,), default=None
            Positions of each round in the given logged bandit feedback.
            If None is given, a learner assumes that there is only one position.
            When `len_list` > 1, position has to be set.
            Currently, this feature is not supported.

        estimated_rewards: array-like, shape (n_rounds, n_actions, len_list), default=None
            Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.
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
        if estimated_rewards_by_reg_model is None:
            estimated_rewards_by_reg_model = np.zeros(
                (context.shape[0], self.n_actions, self.len_list)
            )

        if self.len_list == 1:
            position = np.zeros_like(action, dtype=int)
        else:
            raise NotImplementedError("len_list > 1 is not supported")

        if self.batch_size == "auto":
            batch_size_ = min(200, context.shape[0])
        else:
            batch_size_ = self.batch_size

        context_tensor = torch.from_numpy(context)
        action_tensor = torch.from_numpy(action)
        reward_tensor = torch.from_numpy(reward)
        pscore_tensor = torch.from_numpy(pscore)
        estimated_rewards_by_reg_model_tensor = torch.from_numpy(
            estimated_rewards_by_reg_model
        )

        dataset = torch.util.data.TensorDataset(
            context_tensor,
            action_tensor,
            reward_tensor,
            pscore_tensor,
            estimated_rewards_by_reg_model_tensor,
        )

        if self.early_stopping:
            validation_size = int(context.shape[0] * self.validation_fraction)
            train_size = context.shape[0] - validation_size
            train_dataset, validation_dataset = torch.util.data.random_split(
                dataset, [train_size, validation_size]
            )
            train_loader = torch.util.data.DataLoader(
                train_dataset,
                batch_size=batch_size_,
                shuffle=self.shuffle,
            )
            validation_loader = torch.util.data.DataLoader(
                validation_dataset,
                batch_size=batch_size_,
                shuffle=self.shuffle,
            )
            n_not_improved = 0
        else:
            train_loader = torch.util.data.DataLoader(
                dataset,
                batch_size=batch_size_,
                shuffle=self.shuffle,
            )

        if self.solver in ("adam", "sgd"):
            previous_loss = None
            for epoch in range(self.max_iter):
                self.nn_model.train()
                for bacth_idx, (x, a, r, p, q_hat) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    action_dist = self.nn_model(x)
                    loss = -1.0 * self.objective(
                        reward=r,
                        action=a,
                        pscore=p,
                        action_dist=action_dist,
                        estimated_reward_by_reg_model=q_hat,
                    )
                    loss.backward()
                    self.optimizer.step()

                if self.early_stopping and previous_loss is not None:
                    self.nn_model.eval()
                    for bacth_idx, (x, a, r, p, q_hat) in enumerate(validation_loader):
                        action_dist = self.nn_model(x)
                        loss = -1.0 * self.objective(
                            reward=r,
                            action=a,
                            pscore=p,
                            action_dist=action_dist,
                            estimated_reward_by_reg_model=q_hat,
                        )
                        if loss - previous_loss < self.tol:
                            n_not_improved += 1
                        else:
                            n_not_improved = 0
                        if n_not_improved > self.n_iter_no_change:
                            break
        elif self.solver == "lbfgs":
            for bactch_idx, (x, a, r, p, q_hat) in enumerate(train_loader):

                def closure():
                    self.optimizer.zero_grad()
                    action_dist = self.nn_model(x)
                    loss = -1.0 * self.objective(
                        reward=r,
                        action=a,
                        pscore=p,
                        action_dist=action_dist,
                        estimated_reward_by_reg_model=q_hat,
                    )
                    loss.backward()
                    return loss

                self.optimizer.step(closure)

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
        self.model.eval()
        x = torch.from_numpy(context)
        y = self.nn_model(x)
        predicted_actions_at_position = torch.argmax(y, dim=1).numpy()
        n_rounds = context.shape[0]
        action_dist = np.zeros((n_rounds, self.n_actions, self.len_list))
        action_dist[
            np.arange(n_rounds),
            predicted_actions_at_position,
            np.ones(n_rounds, dtype=int) * position_,
        ] = 1

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

        self.model.eval()
        x = torch.from_numpy(context)
        y = self.nn_model(x)
        return y.numpy()

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
        ), f"predict_proba method can be used only when len_list = 1"
        assert (
            isinstance(context, np.ndarray) and context.ndim == 2
        ), "context must be 2-dimensional ndarray"
        check_scalar(tau, name="tau", target_type=(int, float), min_val=0)

        score_predicted = self.predict_score(context=context)
        choice_prob = softmax(score_predicted / tau, axis=1)
        return choice_prob


def inverse_probability_weighting_tensor(
    reward: torch.Tensor,
    action: np.ndarray,
    pscore: torch.Tensor,
    action_dist: torch.Tensor,
    position: Optional[np.ndarray] = None,
    **kwargs,
) -> torch.Tensor:
    """Estimate policy value of an evaluation policy.

    Parameters
    ----------
    reward: array-like, shape (n_rounds,)
        Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

    action: array-like, shape (n_rounds,)
        Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    pscore: array-like, shape (n_rounds,)
        Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

    position: array-like, shape (n_rounds,), default=None
        Positions of each round in the given logged bandit feedback.

    Returns
    ----------
    V_hat: torch.Tensor
        Estimated policy value (performance) of a given evaluation policy.

    """
    if position is None:
        position = np.zeros(action_dist.shape[0], dtype=int)
    iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
    return (reward * iw).mean()


def self_normalized_inverse_probability_weighting_tensor(
    reward: torch.Tensor,
    action: np.ndarray,
    pscore: torch.Tensor,
    action_dist: torch.Tensor,
    position: Optional[np.ndarray] = None,
    **kwargs,
) -> torch.Tensor:
    """Estimate policy value of an evaluation policy.

    Parameters
    ----------
    reward: torch.Tensor, shape (n_rounds,)
        Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

    action: np.ndarray, shape (n_rounds,)
        Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    pscore: torch.Tensor, shape (n_rounds,)
        Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

    action_dist: torch.Tensor, shape (n_rounds, n_actions, len_list)
        Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

    position: array-like, shape (n_rounds,), default=None
        Positions of each round in the given logged bandit feedback.

    Returns
    ----------
    minus_V_hat: torch.Tensor
        Estimated policy value (performance) of a given evaluation policy.

    """
    if position is None:
        position = np.zeros(action_dist.shape[0], dtype=int)
    iw = action_dist[np.arange(action.shape[0]), action, position] / pscore
    return (reward * iw / iw.mean()).mean()


def directe_method_tensor(
    action_dist: torch.Tensor,
    estimated_rewards_by_reg_model: torch.Tensor,
    position: Optional[np.ndarray] = None,
    **kwargs,
) -> torch.Tensor:
    """Estimate policy value of an evaluation policy.

    Parameters
    ----------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

    estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list)
        Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

    position: array-like, shape (n_rounds,), default=None
        Positions of each round in the given logged bandit feedback.

    Returns
    ----------
    V_hat: torch.Tensor
        Estimated policy value (performance) of a given evaluation policy.

    """
    if position is None:
        position = np.zeros(action_dist.shape[0], dtype=int)
    n_rounds = position.shape[0]
    q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n_rounds), :, position]
    pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
    return torch.mean(q_hat_at_position * pi_e_at_position, dim=1).mean()


def doubly_robust_tensor(
    reward: torch.Tensor,
    action: np.ndarray,
    pscore: torch.Tensor,
    action_dist: torch.Tensor,
    estimated_rewards_by_reg_model: torch.Tensor,
    position: Optional[np.ndarray] = None,
    **kwargs,
) -> torch.Tensor:
    """Estimate policy value of an evaluation policy.

    Parameters
    ----------
    reward: torch.Tensor, shape (n_rounds,)
        Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

    action: array-like, shape (n_rounds,)
        Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    pscore: torch.Tensor, shape (n_rounds,)
        Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

    action_dist: torch.Tensor, shape (n_rounds, n_actions, len_list)
        Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

    estimated_rewards_by_reg_model: torch.Tensor, shape (n_rounds, n_actions, len_list)
        Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

    position: array-like, shape (n_rounds,), default=None
        Positions of each round in the given logged bandit feedback.

    Returns
    ----------
    V_hat: torch.Tensor
        Estimated policy value by the DR estimator.

    """
    if position is None:
        position = np.zeros(action_dist.shape[0], dtype=int)
    n_rounds = action.shape[0]
    iw = action_dist[np.arange(n_rounds), action, position] / pscore
    q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n_rounds), :, position]
    q_hat_factual = estimated_rewards_by_reg_model[
        np.arange(n_rounds), action, position
    ]
    pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
    estimated_rewards = q_hat_at_position * pi_e_at_position
    estimated_rewards = torch.mean(q_hat_at_position * pi_e_at_position, dim=1)
    estimated_rewards += iw * (reward - q_hat_factual)
    return estimated_rewards.mean()


def self_normalized_doubly_robust_tensor(
    reward: torch.Tensor,
    action: np.ndarray,
    pscore: torch.Tensor,
    action_dist: torch.Tensor,
    estimated_rewards_by_reg_model: torch.Tensor,
    position: Optional[np.ndarray] = None,
    **kwargs,
) -> torch.Tensor:
    """Estimate policy value of an evaluation policy.

    Parameters
    ----------
    reward: torch.Tensor, shape (n_rounds,)
        Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

    action: array-like, shape (n_rounds,)
        Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    pscore: torch.Tensor, shape (n_rounds,)
        Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

    action_dist: torch.Tensor, shape (n_rounds, n_actions, len_list)
        Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

    estimated_rewards_by_reg_model: torch.Tensor, shape (n_rounds, n_actions, len_list)
        Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

    position: array-like, shape (n_rounds,), default=None
        Positions of each round in the given logged bandit feedback.

    Returns
    ----------
    V_hat: torch.Tensor
        Estimated policy value by the SNDR estimator.

    """
    if position is None:
        position = np.zeros(action_dist.shape[0], dtype=int)
    n_rounds = action.shape[0]
    iw = action_dist[np.arange(n_rounds), action, position] / pscore
    q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n_rounds), :, position]
    q_hat_factual = estimated_rewards_by_reg_model[
        np.arange(n_rounds), action, position
    ]
    pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
    estimated_rewards = q_hat_at_position * pi_e_at_position
    estimated_rewards = torch.mean(q_hat_at_position * pi_e_at_position, dim=1)
    estimated_rewards += iw * (reward - q_hat_factual) / iw.mean()
    return estimated_rewards.mean()


def switch_doubly_robust_tensor(
    reward: torch.Tensor,
    action: np.ndarray,
    pscore: torch.Tensor,
    action_dist: torch.Tensor,
    estimated_rewards_by_reg_model: torch.Tensor,
    position: Optional[np.ndarray] = None,
    tau: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Estimate policy value of an evaluation policy.

    Parameters
    ----------
    reward: torch.Tensor, shape (n_rounds,)
        Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

    action: array-like, shape (n_rounds,)
        Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    pscore: torch.Tensor, shape (n_rounds,)
        Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

    action_dist: torch.Tensor, shape (n_rounds, n_actions, len_list)
        Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

    estimated_rewards_by_reg_model: torch.Tensor, shape (n_rounds, n_actions, len_list)
        Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

    position: array-like, shape (n_rounds,), default=None
        Positions of each round in the given logged bandit feedback.

    Returns
    ----------
    V_hat: torch.Tensor
        Estimated policy value by the Switch-DR estimator.
    """
    n_rounds = action.shape[0]
    iw = action_dist[np.arange(n_rounds), action, position] / pscore
    switch_indicator = np.array(iw <= tau, dtype=int)
    q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n_rounds), :, position]
    q_hat_factual = estimated_rewards_by_reg_model[
        np.arange(n_rounds), action, position
    ]
    pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
    estimated_rewards = torch.mean(
        q_hat_at_position * pi_e_at_position,
        dim=1,
    )
    estimated_rewards += switch_indicator * iw * (reward - q_hat_factual)
    return estimated_rewards.mean()


def doubly_robust_with_shrinkage_tensor(
    reward: torch.Tensor,
    action: np.ndarray,
    pscore: torch.Tensor,
    action_dist: torch.Tensor,
    estimated_rewards_by_reg_model: torch.Tensor,
    position: Optional[np.ndarray] = None,
    lambda_: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """Estimate policy value of an evaluation policy.

    Parameters
    ----------
    reward: torch.Tensor, shape (n_rounds,)
        Reward observed in each round of the logged bandit feedback, i.e., :math:`r_t`.

    action: array-like, shape (n_rounds,)
        Action sampled by a behavior policy in each round of the logged bandit feedback, i.e., :math:`a_t`.

    pscore: torch.Tensor, shape (n_rounds,)
        Action choice probabilities by a behavior policy (propensity scores), i.e., :math:`\\pi_b(a_t|x_t)`.

    action_dist: torch.Tensor, shape (n_rounds, n_actions, len_list)
        Action choice probabilities by the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_t|x_t)`.

    estimated_rewards_by_reg_model: torch.Tensor, shape (n_rounds, n_actions, len_list)
        Expected rewards for each round, action, and position estimated by a regression model, i.e., :math:`\\hat{q}(x_t,a_t)`.

    position: array-like, shape (n_rounds,), default=None
        Positions of each round in the given logged bandit feedback.

    Returns
    ----------
    V_hat: torch.Tensor
        Estimated policy value by the DRoS estimator.
    """
    n_rounds = action.shape[0]
    iw = action_dist[np.arange(n_rounds), action, position] / pscore
    shrinkage_weight = (lambda_ * iw) / (iw ** 2 + lambda_)
    q_hat_at_position = estimated_rewards_by_reg_model[np.arange(n_rounds), :, position]
    q_hat_factual = estimated_rewards_by_reg_model[
        np.arange(n_rounds), action, position
    ]
    pi_e_at_position = action_dist[np.arange(n_rounds), :, position]
    estimated_rewards = torch.mean(q_hat_at_position * pi_e_at_position, dim=1)
    estimated_rewards += shrinkage_weight * (reward - q_hat_factual)
    return estimated_rewards.mean()
