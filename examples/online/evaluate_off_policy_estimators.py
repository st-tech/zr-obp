import argparse
from pathlib import Path

import numpy as np
from pandas import DataFrame
from joblib import Parallel, delayed

from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
)
from obp.policy import (
    BernoulliTS,
    EpsilonGreedy,
    LinEpsilonGreedy,
    LinTS,
    LinUCB,
    LogisticEpsilonGreedy,
    LogisticTS,
    LogisticUCB,
)
from obp.ope import OffPolicyEvaluation, ReplayMethod
from obp.simulator import calc_ground_truth_policy_value, run_bandit_simulation


ope_estimators = [ReplayMethod()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate off-policy estimators with online bandit algorithms and synthetic bandit data."
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="number of simulations in the experiment."
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=10000,
        help="number of rounds for synthetic bandit feedback.",
    )
    parser.add_argument(
        "--n_actions",
        type=int,
        default=10,
        help="number of actions for synthetic bandit feedback.",
    )
    parser.add_argument(
        "--dim_context",
        type=int,
        default=5,
        help="dimensions of context vectors characterizing each round.",
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
        action_dist = run_bandit_simulation(
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
        relative_ee_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=ground_truth_policy_value,
            action_dist=action_dist,
        )

        return relative_ee_i

    processed = Parallel(
        n_jobs=n_jobs,
        verbose=50,
    )([delayed(process)(i) for i in np.arange(n_runs)])
    relative_ee_dict = {est.estimator_name: dict() for est in ope_estimators}
    for i, relative_ee_i in enumerate(processed):
        for (
            estimator_name,
            relative_ee_,
        ) in relative_ee_i.items():
            relative_ee_dict[estimator_name][i] = relative_ee_
    relative_ee_df = DataFrame(relative_ee_dict).describe().T.round(6)

    print("=" * 45)
    print(f"random_state={random_state}")
    print("-" * 45)
    print(relative_ee_df[["mean", "std"]])
    print("=" * 45)

    # save results of the evaluation of off-policy estimators in './logs' directory.
    log_path = Path("./logs")
    log_path.mkdir(exist_ok=True, parents=True)
    relative_ee_df.to_csv(log_path / "relative_ee_of_ope_estimators.csv")
