# Examples with Synthetic Data
Here, we use synthetic bandit dataset and pipeline to evaluate OPE estimators.


## Running experiments

**Evaluating Off-Policy Estimators**

We evaluate the estimation performances of Direct Method (DM), Inverse Probability Weighting (IPW), Self-Normalized Inverse Probability Weighting (SNIPW), Doubly Robust (DR), Self-Normalized Doubly Robust (SNDR), and Switch Doubly Robust (Switch-DR).
[`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators using the synthetic bandit feedback data.

```bash
# run evaluation of OPE estimators.
python evaluate_off_policy_estimators.py\
    --n_runs $n_runs\
    --n_rounds $n_rounds\
    --n_actions $n_actions\
    --dim_context $dim_context\
    --dim_action_context $dim_action_context\
    --counterfactual_policy $counterfactual_policy\
    --random_state $random_state
```
where `$n_runs` specifies the number of simulation runs in the experiment to estimate confidence intervals of the performance of OPE estimators.
`$n_rounds` and `$n_actions` specify the number of rounds and the number of actions of the synthetic bandit data.
`$dim_context` and `$dim_action_context` specify the dimension of context vectors characterizing each round and action, respectively.
`$counterfactual_policy` specifies the counterfactual policy.
It should be one of 'bts', 'random', 'linear_ts', 'linear_ucb', 'linear_egreedy', 'logistic_ts', 'logistic_ucb', and 'logistic_egreedy'.

For example, the following command compares the estimation performances of the OPE estimators using the synthetic bandit feedback data with 100,000 rounds, 10 actions, context vectors with five dimensions.

```bash
python evaluate_off_policy_estimators.py\
    --n_runs 10\
    --n_rounds 100000\
    --n_actions 10\
    --dim_context 5\
    --dim_action_context 5\
    --counterfactual_policy linear_egreedy\
    --random_state 12345

# relative estimation errors (lower is better) and their 95% confidence intervals of OPE estimators.
# our evaluation of OPE procedure suggests that IPW performs better than the other model dependent estimators such as DM and DR.
# ============================================================
# random_state=12345
# ------------------------------------------------------------
#                mean  95.0% CI (lower)  95.0% CI (upper)
# dm         0.002284          0.001167          0.003887
# ipw        0.006241          0.003920          0.008733
# snipw      0.011473          0.008055          0.014634
# dr         0.007346          0.004698          0.009996
# sndr       0.016752          0.010225          0.023883
# switch-dr  0.005359          0.003168          0.007579
# ============================================================
```

The above result can change with different situations.
You can try the evaluation of OPE with other experimental settings easily!
