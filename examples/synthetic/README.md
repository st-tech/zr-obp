# Examples with Synthetic Data
Here, we use synthetic bandit dataset and pipeline to evaluate OPE estimators.


## Running experiments

**Example Experimet. Evaluating Off-Policy Estimators**

We evaluate the estimation performance of Direct Method (DM), Inverse Probability Weighting (IPW), Self-Normalized Inverse Probability Weighting (SNIPW), Doubly Robust (DR), Self-Normalized Doubly Robust (SNDR), and Switch Doubly Robust (Switch-DR).
[`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators.

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
`$n_rounds` and `$n_actions` specify the number of rounds and the number of actions for the synthetic bandit data.
`$dim_context` and `$dim_action_context` specify the dimension of context vectors characterizing each round and action, respectively.
`$counterfactual_policy` specifies the counterfactual policy.
They should be one of 'bts', 'random', 'logistic_ts', 'logistic_ucb', and 'logistic_egreedy'.

For example, the following command compares the estimation performance of the OPE estimators by synthetic bandit feedback data with 100,000 rounds, 20 actions, context vectors with five dimensions.

```bash
python evaluate_off_policy_estimators.py\
    --n_runs 10\
    --n_rounds 100000\
    --n_actions 20\
    --dim_context 5\
    --dim_action_context 5\
    --counterfactual_policy logistic_ts\
    --random_state 12345

# relative estimation errors (lower is better) and their 95% confidence intervals of OPE estimators.
# our evaluation of OPE procedure suggests that IPW and SNIPW perform better than other
# model dependent estimators including DM, DR, SNDR, and Switch-DR.
# ============================================================
# random_state=12345
# ------------------------------------------------------------
#                mean  95.0% CI (lower)  95.0% CI (upper)
# dm         0.084652           0.05387           0.11658
# ipw        0.023547           0.01196           0.03720
# snipw      0.031046           0.01254           0.05261
# dr         0.034639           0.02392           0.04739
# sndr       0.065582           0.03284           0.10466
# switch-dr  0.050021           0.03090           0.07106
# ============================================================
```

Let's try the evaluation of OPE with other experimental settings!
