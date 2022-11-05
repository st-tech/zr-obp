import argparse
from pathlib import Path

from joblib import delayed
from joblib import Parallel
import numpy as np
from pandas import DataFrame

from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticBanditDataset
from obp.ope import OffPolicyEvaluation
from obp.ope import ReplayMethod
from obp.policy import BernoulliTS
from obp.policy import EpsilonGreedy
from obp.policy import LinEpsilonGreedy
from obp.policy import LinTS
from obp.policy import LinUCB
from obp.policy import LogisticEpsilonGreedy
from obp.policy import LogisticTS
from obp.policy import LogisticUCB
from obp.simulator import calc_ground_truth_policy_value
from obp.utils import run_bandit_replay

ope_estimators = [ReplayMethod()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate off-policy estimators with replay bandit algorithms and synthetic bandit data."
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="number of simulations in the experiment."
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=10000,
        help="sample size of logged bandit data.",
    )
    parser.add_argument(
        "--n_actions",
        type=int,
        default=10,
        help="number of actions.",
    )
    parser.add_argument(
        "--dim_context",
        type=int,
        default=5,
        help="dimensions of context vectors.",
    )
    parser.add_argument(
        "--n_sim",
        type=int,
        default=1,
        help="number of simulations to calculate ground truth policy values",
    )
    parser.add_argument(
        "--evaluation_policy_name",
        type=str,
        choices=[
            "bernoulli_ts",
            "epsilon_greedy",
            "lin_epsilon_greedy",
            "lin_ts",
            "lin_ucb",
            "logistic_epsilon_greedy",
            "logistic_ts",
            "logistic_ucb",
        ],
        required=True,
        help="the name of evaluation policy, bernoulli_ts, epsilon_greedy, lin_epsilon_greedy, lin_ts, lin_ucb, logistic_epsilon_greedy, logistic_ts, or logistic_ucb",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="the maximum number of concurrently running jobs.",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations
    n_runs = args.n_runs
    n_rounds = args.n_rounds
    n_actions = args.n_actions
    dim_context = args.dim_context
    n_sim = args.n_sim
    evaluation_policy_name = args.evaluation_policy_name
    n_jobs = args.n_jobs
    random_state = args.random_state
    np.random.seed(random_state)

    # define evaluation policy
    evaluation_policy_dict = dict(
        bernoulli_ts=BernoulliTS(n_actions=n_actions, random_state=random_state),
        epsilon_greedy=EpsilonGreedy(
            n_actions=n_actions, epsilon=0.1, random_state=random_state
        ),
        lin_epsilon_greedy=LinEpsilonGreedy(
            dim=dim_context, n_actions=n_actions, epsilon=0.1, random_state=random_state
        ),
        lin_ts=LinTS(dim=dim_context, n_actions=n_actions, random_state=random_state),
        lin_ucb=LinUCB(dim=dim_context, n_actions=n_actions, random_state=random_state),
        logistic_epsilon_greedy=LogisticEpsilonGreedy(
            dim=dim_context, n_actions=n_actions, epsilon=0.1, random_state=random_state
        ),
        logistic_ts=LogisticTS(
            dim=dim_context, n_actions=n_actions, random_state=random_state
        ),
        logistic_ucb=LogisticUCB(
            dim=dim_context, n_actions=n_actions, random_state=random_state
        ),
    )
    evaluation_policy = evaluation_policy_dict[evaluation_policy_name]

    def process(i: int):
        # synthetic data generator with uniformly random policy
        dataset = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            reward_function=logistic_reward_function,
            behavior_policy_function=None,  # uniformly random
            random_state=i,
        )
        # sample new data of synthetic logged bandit feedback
        bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        # simulate the evaluation policy
        action_dist = run_bandit_replay(
            bandit_feedback=bandit_feedback, policy=evaluation_policy
        )
        # estimate the ground-truth policy values of the evaluation policy
        # by Monte-Carlo Simulation using p(r|x,a), the reward distribution
        ground_truth_policy_value = calc_ground_truth_policy_value(
            bandit_feedback=bandit_feedback,
            reward_sampler=dataset.sample_reward,  # p(r|x,a)
            policy=evaluation_policy,
            n_sim=n_sim,  # the number of simulations
        )
        # evaluate estimators' performances using relative estimation error (relative-ee)
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback,
            ope_estimators=ope_estimators,
        )
        metric_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            action_dist=action_dist,
        )

        return metric_i

    processed = Parallel(
        n_jobs=n_jobs,
        verbose=50,
    )([delayed(process)(i) for i in np.arange(n_runs)])
    metric_dict = {est.estimator_name: dict() for est in ope_estimators}
    for i, metric_i in enumerate(processed):
        for (
            estimator_name,
            relative_ee_,
        ) in metric_i.items():
            metric_dict[estimator_name][i] = relative_ee_
    se_df = DataFrame(metric_dict).describe().T.round(6)

    print("=" * 45)
    print(f"random_state={random_state}")
    print("-" * 45)
    print(se_df[["mean", "std"]])
    print("=" * 45)

    # save results of the evaluation of off-policy estimators in './logs' directory.
    log_path = Path("./logs")
    log_path.mkdir(exist_ok=True, parents=True)
    se_df.to_csv(log_path / "relative_ee_of_ope_estimators.csv")
