# Example with Online Bandit Algorithms


## Description

Here, we use synthetic bandit datasets to evaluate OPE of online bandit algorithms.
Specifically, we evaluate the estimation performances of well-known off-policy estimators using the ground-truth policy value of an evaluation policy calculable with synthetic data.


## Evaluating Off-Policy Estimators

In the following, we evaluate the estimation performances of Replay Method (RM).
RM uses a subset of the logged bandit feedback data where actions selected by the behavior policy are the same as that of the evaluation policy.
Theoretically, RM is unbiased when the behavior policy is uniformly random and the evaluation policy is fixed.
However, empirically, RM works well when evaluation policies are learning algorithms.
Please refer to https://arxiv.org/abs/1003.5956 about the details of RM.


### Files
- [`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators by RM using synthetic bandit feedback data.

### Scripts

```bash
# run evaluation of OPE estimators with synthetic bandit data
python evaluate_off_policy_estimators.py\
    --n_runs $n_runs\
    --n_rounds $n_rounds\
    --n_actions $n_actions\
    --n_sim $n_sim\
    --dim_context $dim_context\
    --n_jobs $n_jobs\
    --random_state $random_state
```
- `$n_runs` specifies the number of simulation runs in the experiment to estimate standard deviations of the performance of OPE estimators.
- `$n_rounds` and `$n_actions` specify the number of rounds (or samples) and the number of actions of the synthetic bandit data.
- `$dim_context` specifies the dimension of context vectors.
- `$n_sim` specifeis the simulations in the Monte Carlo simulation to compute the ground-truth policy value.
- `$evaluation_policy_name` specifeis the evaluation policy and should be one of "bernoulli_ts", "epsilon_greedy", "lin_epsilon_greedy", "lin_ts, lin_ucb", "logistic_epsilon_greedy", "logistic_ts", or "logistic_ucb".
- `$n_jobs` is the maximum number of concurrently running jobs.

For example, the following command compares the estimation performances (relative estimation error; relative-ee) of the OPE estimators using the synthetic bandit feedback data with 100,000 rounds, 30 actions, five dimensional context vectors.

```bash
python evaluate_off_policy_estimators.py\
    --n_runs 20\
    --n_rounds 1000\
    --n_actions 30\
    --dim_context 5\
    --evaluation_policy_name bernoulli_ts\
    --n_sim 3\
    --n_jobs -1\
    --random_state 12345

# relative-ee of OPE estimators and their standard deviations (lower means accurate).
# =============================================
# random_state=12345
# ---------------------------------------------
#         mean       std
# rm  0.097064  0.091453
# =============================================
```

The above result can change with different situations.
You can try the evaluation of OPE with other experimental settings easily.
