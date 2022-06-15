from dataclasses import dataclass
import itertools
from typing import Dict
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_scalar

from obp.utils import check_array
from obp.utils import check_ope_inputs

from ..utils import estimate_confidence_interval_by_bootstrap
from .estimators import BaseOffPolicyEstimator


@dataclass
class MarginalizedInverseProbabilityWeighting(BaseOffPolicyEstimator):
    """Marginalized Inverse Probability Weighting (MIPW) Estimator.

    Note
    -------
    MIPW estimates the policy value of the evaluation policy :math:`\\pi_e` as

    .. math::
        \\hat{V}_{\\mathrm{MIPW}} (\\pi_e; \\mathcal{D}) := \\mathbb{E}_{n} [ w(x_i,e_i) r_i],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,e_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations
    collected by behavior policy :math:`\\pi_b`.
    :math:`w(x,e):=p (e|x, \\pi_e)/p (e|x, \\pi_b)` is the marginal importance weight given context :math:`x`
    and action embedding :math:`e`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    MIPW leverages the action embeddings such as item category information to reduce the variance of IPW under assumptions different from the vanilla IPW.
    This estimator is proposed mainly to deal with large discrete action spaces, which are everywhere in industrial applications.
    Note that in the reference paper, MIPW is called the MIPS estimator.

    Parameters
    ------------
    n_actions: int
        Number of actions in the logged data.

    pi_a_x_e_estimator: ClassifierMixin, default=`sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=12345)`
        A sklearn classifier to estimate :math:`\\pi(a|x,e)`.
        It is then used to estimate the marginal importance weight as
        :math:`\\hat{w}(x,e) = \\mathbb{E}_{\\hat{\\pi}(a|x,e)}[w(x,a)]`.

    embedding_selection_method, default=None
        Method to conduct data-driven action embedding selection. Must be one of None, 'exact', or 'greedy'.
        If None, the given action embedding (action context) will be used to estimate the marginal importance weights.
        If 'greedy', a greedy version of embedding selection will be applied, which is significantly faster than 'exact',
        but might be worse in terms of OPE performance.
        If the number of action embedding dimensions is larger than 20, 'greedy' is a recommended choice.

    min_emb_dim: int, default=1
        Minimum number of action embedding dimensions to be used in estimating the marginal importance weights.

    delta: float, default=0.05
        Confidence level used to estimate the deviation bound in data-driven action embedding selection.

    estimator_name: str, default='mipw'.
        Name of the estimator.

    References
    ------------
    Yuta Saito and Thorsten Joachims.
    "Off-Policy Evaluation for Large Action Spaces via Embeddings." 2022.

    """

    n_actions: int
    pi_a_x_e_estimator: ClassifierMixin = LogisticRegression(
        max_iter=1000, random_state=12345
    )
    embedding_selection_method: Optional[str] = None
    min_emb_dim: int = 1
    delta: float = 0.05
    estimator_name: str = "mipw"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, name="n_actions", target_type=int, min_val=2)
        check_scalar(
            self.min_emb_dim,
            name="min_emb_dim",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            self.delta,
            name="delta",
            target_type=float,
            min_val=0.0,
            max_val=1.0,
        )
        if self.embedding_selection_method is not None:
            if self.embedding_selection_method not in ["exact", "greedy"]:
                raise ValueError(
                    "If given, `embedding_selection_method` must be either 'exact' or 'greedy', but"
                    f"{self.embedding_selection_method} is given."
                )
        if not is_classifier(self.pi_a_x_e_estimator):
            raise ValueError("`pi_a_x_e_estimator` must be a classifier.")

    def _estimate_round_rewards(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        p_e_a: Optional[np.ndarray] = None,
        with_dev: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_embed: array-like, shape (n_rounds, dim_action_embed)
            Context vectors characterizing actions or action embeddings such as item category information.
            This is used to estimate the marginal importance weights.

        pi_b: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the logging/behavior policy, i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        p_e_a: array-like, shape (n_actions, n_cat_per_dim, n_cat_dim), default=None
            Conditional distribution of action embeddings given each action.
            This distribution is available only when we use synthetic bandit data, i.e.,
            `obp.dataset.SyntheticBanditDatasetWithActionEmbeds`.
            See the output of the `obtain_batch_bandit_feedback` argument of this class.
            If `p_e_a` is given, MIPW uses the true marginal importance weights based on this distribution.
            The performance of MIPW with the true weights is useful in synthetic experiments of research papers.

        with_dev: bool, default=False.
            Whether to output a deviation bound with the estimated sample-wise rewards.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        n = reward.shape[0]
        if p_e_a is not None:
            p_e_pi_b = np.ones(n)
            p_e_pi_e = np.ones(n)
            for d in np.arange(p_e_a.shape[-1]):
                p_e_pi_b_d = pi_b[np.arange(n), :, position] @ p_e_a[:, :, d]
                p_e_pi_b *= p_e_pi_b_d[np.arange(n), action_embed[:, d]]
                p_e_pi_e_d = action_dist[np.arange(n), :, position] @ p_e_a[:, :, d]
                p_e_pi_e *= p_e_pi_e_d[np.arange(n), action_embed[:, d]]
            w_x_e = p_e_pi_e / p_e_pi_b
            self.max_w_x_e = w_x_e.max()

        else:
            w_x_e = self._estimate_w_x_e(
                context=context,
                action=action,
                action_embed=action_embed,
                pi_e=action_dist[np.arange(n), :, position],
                pi_b=pi_b[np.arange(n), :, position],
            )
            self.max_w_x_e = w_x_e.max()

        if with_dev:
            r_hat = reward * w_x_e
            cnf = np.sqrt(np.var(r_hat) / (n - 1))
            cnf *= stats.t.ppf(1.0 - (self.delta / 2), n - 1)

            return r_hat.mean(), cnf

        return reward * w_x_e

    def _estimate_w_x_e(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        pi_e: np.ndarray,
    ) -> np.ndarray:
        """Estimate the marginal importance weights."""
        n = action.shape[0]
        w_x_a = pi_e / pi_b
        w_x_a = np.where(w_x_a < np.inf, w_x_a, 0)
        c = OneHotEncoder(
            sparse=False,
            drop="first",
        ).fit_transform(action_embed)
        x_e = np.c_[context, c]
        pi_a_x_e = np.zeros((n, self.n_actions))
        self.pi_a_x_e_estimator.fit(x_e, action)
        pi_a_x_e[:, np.unique(action)] = self.pi_a_x_e_estimator.predict_proba(x_e)
        w_x_e = (w_x_a * pi_a_x_e).sum(1)

        return w_x_e

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        action_dist: np.ndarray,
        context: Optional[np.ndarray] = None,
        p_e_a: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_embed: array-like, shape (n_rounds, dim_action_embed)
            Context vectors characterizing actions or action embeddings such as item category information.
            This is used to estimate the marginal importance weights.

        pi_b: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the logging/behavior policy, i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        context: array-like, shape (n_rounds, dim_context), default=None
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        p_e_a: array-like, shape (n_actions, n_cat_per_dim, n_cat_dim), default=None
            Conditional distribution of action embeddings given each action.
            This distribution is available only when we use synthetic bandit data, i.e.,
            `obp.dataset.SyntheticBanditDatasetWithActionEmbeds`.
            See the output of the `obtain_batch_bandit_feedback` argument of this class.
            If `p_e_a` is given, MIPW uses the true marginal importance weights based on this distribution.
            The performance of MIPW with the true weights is useful in synthetic experiments of research papers.

        Returns
        ----------
        V_hat: float
            Estimated policy value of the evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action_embed, name="action_embed", expected_dim=2)
        check_array(array=pi_b, name="pi_b", expected_dim=3)
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        check_ope_inputs(
            action_dist=pi_b,
            position=position,
            action=action,
            reward=reward,
        )
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        if p_e_a is not None:
            check_array(array=p_e_a, name="p_e_a", expected_dim=3)
        else:
            check_array(array=context, name="context", expected_dim=2)

        if action_embed.shape[1] > 1 and self.embedding_selection_method is not None:
            if self.embedding_selection_method == "exact":
                return self._estimate_with_exact_pruning(
                    context=context,
                    reward=reward,
                    action=action,
                    action_embed=action_embed,
                    position=position,
                    pi_b=pi_b,
                    action_dist=action_dist,
                    p_e_a=p_e_a,
                )
            elif self.embedding_selection_method == "greedy":
                return self._estimate_with_greedy_pruning(
                    context=context,
                    reward=reward,
                    action=action,
                    action_embed=action_embed,
                    position=position,
                    pi_b=pi_b,
                    action_dist=action_dist,
                    p_e_a=p_e_a,
                )
        else:
            return self._estimate_round_rewards(
                context=context,
                reward=reward,
                action=action,
                action_embed=action_embed,
                position=position,
                pi_b=pi_b,
                action_dist=action_dist,
                p_e_a=p_e_a,
            ).mean()

    def _estimate_with_exact_pruning(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        action_dist: np.ndarray,
        position: np.ndarray,
        p_e_a: Optional[np.ndarray] = None,
    ) -> float:
        """Apply an exact version of data-drive action embedding selection."""
        n_emb_dim = action_embed.shape[1]
        theta_list, cnf_list = [], []
        feat_list, C = np.arange(n_emb_dim), np.sqrt(6) - 1
        for i in np.arange(n_emb_dim, self.min_emb_dim - 1, -1):
            comb_list = list(itertools.combinations(feat_list, i))
            theta_list_, cnf_list_ = [], []
            for comb in comb_list:
                if p_e_a is None:
                    theta, cnf = self._estimate_round_rewards(
                        context=context,
                        reward=reward,
                        action=action,
                        action_embed=action_embed[:, comb],
                        pi_b=pi_b,
                        action_dist=action_dist,
                        position=position,
                        with_dev=True,
                    )
                else:
                    theta, cnf = self._estimate_round_rewards(
                        context=context,
                        reward=reward,
                        action=action,
                        action_embed=action_embed[:, comb],
                        pi_b=pi_b,
                        action_dist=action_dist,
                        position=position,
                        p_e_a=p_e_a[:, :, comb],
                        with_dev=True,
                    )
                if len(theta_list) > 0:
                    theta_list_.append(theta), cnf_list_.append(cnf)
                else:
                    theta_list.append(theta), cnf_list.append(cnf)
                    continue

            idx_list = np.argsort(cnf_list_)[::-1]
            for idx in idx_list:
                theta_i, cnf_i = theta_list_[idx], cnf_list_[idx]
                theta_j, cnf_j = np.array(theta_list), np.array(cnf_list)
                if (np.abs(theta_j - theta_i) <= cnf_i + C * cnf_j).all():
                    theta_list.append(theta_i), cnf_list.append(cnf_i)
                else:
                    return theta_j[-1]

        return theta_j[-1]

    def _estimate_with_greedy_pruning(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        action_dist: np.ndarray,
        position: np.ndarray,
        p_e_a: Optional[np.ndarray] = None,
    ) -> float:
        """Apply a greedy version of data-drive action embedding selection."""
        n_emb_dim = action_embed.shape[1]
        theta_list, cnf_list = [], []
        current_feat, C = np.arange(n_emb_dim), np.sqrt(6) - 1

        # init
        if p_e_a is None:
            theta, cnf = self._estimate_round_rewards(
                context=context,
                reward=reward,
                action=action,
                action_embed=action_embed[:, current_feat],
                pi_b=pi_b,
                action_dist=action_dist,
                position=position,
                with_dev=True,
            )
        else:
            theta, cnf = self._estimate_round_rewards(
                context=context,
                reward=reward,
                action=action,
                action_embed=action_embed[:, current_feat],
                pi_b=pi_b,
                action_dist=action_dist,
                position=position,
                p_e_a=p_e_a[:, :, current_feat],
                with_dev=True,
            )
        theta_list.append(theta), cnf_list.append(cnf)

        # iterate
        while current_feat.shape[0] > self.min_emb_dim:
            theta_list_, cnf_list_, d_list_ = [], [], []
            for d in current_feat:
                idx_without_d = np.where(current_feat != d, True, False)
                candidate_feat = current_feat[idx_without_d]
                if p_e_a is None:
                    theta, cnf = self._estimate_round_rewards(
                        context=context,
                        reward=reward,
                        action=action,
                        action_embed=action_embed[:, candidate_feat],
                        pi_b=pi_b,
                        action_dist=action_dist,
                        position=position,
                        with_dev=True,
                    )
                else:
                    theta, cnf = self._estimate_round_rewards(
                        context=context,
                        reward=reward,
                        action=action,
                        action_embed=action_embed[:, candidate_feat],
                        pi_b=pi_b,
                        action_dist=action_dist,
                        position=position,
                        p_e_a=p_e_a[:, :, candidate_feat],
                        with_dev=True,
                    )
                d_list_.append(d)
                theta_list_.append(theta), cnf_list_.append(cnf)

            idx_list = np.argsort(cnf_list_)[::-1]
            for idx in idx_list:
                excluded_dim = d_list_[idx]
                theta_i, cnf_i = theta_list_[idx], cnf_list_[idx]
                theta_j, cnf_j = np.array(theta_list), np.array(cnf_list)
                if (np.abs(theta_j - theta_i) <= cnf_i + C * cnf_j).all():
                    theta_list.append(theta_i), cnf_list.append(cnf_i)
                else:
                    return theta_j[-1]
            idx_without_d = np.where(current_feat != excluded_dim, True, False)
            current_feat = current_feat[idx_without_d]

        return theta_j[-1]

    def estimate_interval(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        action_dist: np.ndarray,
        context: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        p_e_a: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using nonparametric bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_embed: array-like, shape (n_rounds, dim_action_embed)
            Context vectors characterizing actions or action embeddings such as item category information.
            This is used to estimate the marginal importance weights.

        pi_b: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the logging/behavior policy, i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        context: array-like, shape (n_rounds, dim_context), default=None
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        p_e_a: array-like, shape (n_actions, n_cat_per_dim, n_cat_dim), default=None
            Conditional distribution of action embeddings given each action.
            This distribution is available only when we use synthetic bandit data, i.e.,
            `obp.dataset.SyntheticBanditDatasetWithActionEmbeds`.
            See the output of the `obtain_batch_bandit_feedback` argument of this class.
            If `p_e_a` is given, MIPW uses the true marginal importance weights based on this distribution.
            The performance of MIPW with the true weights is useful in synthetic experiments of research papers.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(array=action_embed, name="action_embed", expected_dim=2)
        check_array(array=pi_b, name="pi_b", expected_dim=3)
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        check_ope_inputs(
            action_dist=pi_b,
            position=position,
            action=action,
            reward=reward,
        )
        check_ope_inputs(
            action_dist=action_dist,
            position=position,
            action=action,
            reward=reward,
        )
        if position is None:
            position = np.zeros(action_dist.shape[0], dtype=int)
        if p_e_a is not None:
            check_array(array=p_e_a, name="p_e_a", expected_dim=3)
        else:
            check_array(array=context, name="context", expected_dim=2)

        estimated_round_rewards = self._estimate_round_rewards(
            context=context,
            reward=reward,
            action=action,
            action_embed=action_embed,
            position=position,
            pi_b=pi_b,
            action_dist=action_dist,
            p_e_a=p_e_a,
        )
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class SelfNormalizedMarginalizedInverseProbabilityWeighting(
    MarginalizedInverseProbabilityWeighting
):
    """Self Normalized Marginalized Inverse Probability Weighting (SNMIPW) Estimator.

    Note
    -------
    SNMIPW is the self-normalized version of MIPW and estimates the policy value of the evaluation policy :math:`\\pi_e` as

    .. math::
        \\hat{V}_{\\mathrm{MIPW}} (\\pi_e; \\mathcal{D}) := \\frac{\\mathbb{E}_{n} [ w(x_i,e_i) r_i]}{\\mathbb{E}_{n} [ w(x_i,e_i)]},

    where :math:`\\mathcal{D}=\\{(x_i,a_i,e_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations
    collected by behavior policy :math:`\\pi_b`.
    :math:`w(x,e):=p (e|x, \\pi_e)/p (e|x, \\pi_b)` is the marginal importance weight given context :math:`x`
    and action embedding :math:`e`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    Parameters
    ------------
    n_actions: int
        Number of actions in the logged data.

    embedding_selection_method, default=None
        Method to conduct data-driven action embedding selection. Must be one of None, 'exact', or 'greedy'.
        If None, the given action embedding (action context) will be used to estimate the marginal importance weights.
        If 'greed', a greedy version of embedding selection will be applied, which is significantly faster than 'exact',
        but might be worse in terms of OPE performance.
        If the number of action embedding dimensions is larger than 20, 'greedy' is a recommended choice.

    min_emb_dim: int, default=1
        Minimum number of action embedding dimensions to be used in estimating the marginal importance weights.

    delta: float, default=0.05
        Confidence level used to estimate the deviation bound in data-driven action embedding selection.

    estimator_name: str, default='snmipw'.
        Name of the estimator.

    References
    ------------
    Yuta Saito and Thorsten Joachims.
    "Off-Policy Evaluation for Large Action Spaces via Embeddings." 2022.

    """

    estimator_name: str = "snmipw"

    def _estimate_round_rewards(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        p_e_a: Optional[np.ndarray] = None,
        with_dev: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_embed: array-like, shape (n_rounds, dim_action_embed)
            Context vectors characterizing actions or action embeddings such as item category information.
            This is used to estimate the marginal importance weights.

        pi_b: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the logging/behavior policy, i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        p_e_a: array-like, shape (n_actions, n_cat_per_dim, n_cat_dim), default=None
            Conditional distribution of action embeddings given each action.
            This distribution is available only when we use synthetic bandit data, i.e.,
            `obp.dataset.SyntheticBanditDatasetWithActionEmbeds`.
            See the output of the `obtain_batch_bandit_feedback` argument of this class.
            If `p_e_a` is given, MIPW uses the true marginal importance weights based on this distribution.
            The performance of MIPW with the true weights is useful in synthetic experiments of research papers.

        with_dev: bool, default=False.
            Whether to output a deviation bound with the estimated sample-wise rewards.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        n = reward.shape[0]
        if p_e_a is not None:
            p_e_pi_b = np.ones(n)
            p_e_pi_e = np.ones(n)
            for d in np.arange(p_e_a.shape[-1]):
                p_e_pi_b_d = pi_b[np.arange(n), :, position] @ p_e_a[:, :, d]
                p_e_pi_b *= p_e_pi_b_d[np.arange(n), action_embed[:, d]]
                p_e_pi_e_d = action_dist[np.arange(n), :, position] @ p_e_a[:, :, d]
                p_e_pi_e *= p_e_pi_e_d[np.arange(n), action_embed[:, d]]
            w_x_e = p_e_pi_e / p_e_pi_b
            self.max_w_x_e = w_x_e.max()

        else:
            w_x_e = self._estimate_w_x_e(
                context=context,
                action=action,
                action_embed=action_embed,
                pi_e=action_dist[np.arange(n), :, position],
                pi_b=pi_b[np.arange(n), :, position],
            )
            self.max_w_x_e = w_x_e.max()

        if with_dev:
            r_hat = reward * w_x_e
            cnf = np.sqrt(np.var(r_hat) / (n - 1))
            cnf *= stats.t.ppf(1.0 - (self.delta / 2), n - 1)

            return r_hat.mean(), cnf

        return reward * w_x_e / w_x_e.mean()
