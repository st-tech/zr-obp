# Examples with Synthetic Data


## Description

Here, we use synthetic bandit dataset and pipeline to evaluate OPE estimators.
Specifically, we evaluate the estimation performances of well-known off-policy estimators using the ground-truth policy value of an evaluation policy, which is calculable with synthetic data.

## Evaluating Off-Policy Estimators

In the following, we evaluate the estimation performances of Direct Method (DM), Inverse Probability Weighting (IPW), Self-Normalized Inverse Probability Weighting (SNIPW), Doubly Robust (DR), Self-Normalized Doubly Robust (SNDR), and Switch Doubly Robust (Switch-DR).

[`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators using the synthetic bandit feedback data.

```bash
# run evaluation of OPE estimators.
python evaluate_off_policy_estimators.py\
    --n_runs $n_runs\
    --n_rounds $n_rounds\
    --n_actions $n_actions\
    --dim_context $dim_context\
    --dim_action_context $dim_action_context\
    --base_model_for_evaluation_policy $base_model_for_evaluation_policy\
    --base_model_for_reg_model $base_model_for_reg_model\
    --random_state $random_state
```
where `$n_runs` specifies the number of simulation runs in the experiment to estimate confidence intervals of the performance of OPE estimators.
`$n_rounds` and `$n_actions` specify the number of rounds and the number of actions of the synthetic bandit data.
`$dim_context` and `$dim_action_context` specify the dimension of context vectors characterizing each round and action, respectively.
`$base_model_for_evaluation_policy` specifies the base ML model for defining evaluation policy and should be one of logistic_regression, random_forest or lightgbm.
`$base_model_for_reg_model` specifies the base ML model for defining regression model and should be one of logistic_regression, random_forest or lightgbm.
It should be one of 'bts', 'random', 'linear_ts', 'linear_ucb', 'linear_egreedy', 'logistic_ts', 'logistic_ucb', and 'logistic_egreedy'.

For example, the following command compares the estimation performances of the OPE estimators using the synthetic bandit feedback data with 100,000 rounds, 30 actions, context vectors with five dimensions.

```bash
python evaluate_off_policy_estimators.py\
    --n_runs 20\
    --n_rounds 100000\
    --n_actions 30\
    --dim_context 5\
    --dim_action_context 5\
    --base_model_for_evaluation_policy logistic_regression\
    --base_model_for_reg_model logistic_regression\
    --random_state 12345

# relative estimation errors (lower is better) and their standard deviations of OPE estimators.
# our evaluation of OPE procedure suggests that Switch-DR performs better than the other estimators.
# ==============================
# random_state=12345
# ------------------------------
#                mean       std
# dm         0.025025  0.005871
# ipw        0.011111  0.016634
# snipw      0.010181  0.007922
# dr         0.008184  0.007690
# sndr       0.011609  0.007727
# switch-dr  0.004839  0.004315
# ==============================
```

The above result can change with different situations.
You can try the evaluation of OPE with other experimental settings easily!
