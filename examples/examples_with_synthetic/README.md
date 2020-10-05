# Examples with Synthetic Data


## Description

Here, we use synthetic bandit dataset and pipeline to evaluate OPE estimators.
Specifically, we evaluate the estimation performances of well-known off-policy estimators using the ground-truth policy value of an evaluation policy, which is calculable with synthetic data.

## Evaluating Off-Policy Estimators

In the following, we evaluate the estimation performances of Direct Method (DM), Inverse Probability Weighting (IPW), Self-Normalized Inverse Probability Weighting (SNIPW), Doubly Robust (DR), Self-Normalized Doubly Robust (SNDR), Switch Inverse Probability Weighting (Switch-IPW), Switch Doubly Robust (Switch-DR), and Doubly Robust with Optimistic Shrinkage (DRos).
For Switch-IPW, Switch-DR, and DRos, we tried some different values of hyperparameters.

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
# our evaluation of OPE procedure suggests that Switch-DR(tau=100) performs better than the other estimators.
# Moreover, it appears that the performance of OPE estimators depend on the choice of hyperparameters.
# ========================================
# random_state=12345
# ----------------------------------------
#                           mean       std
# dm                    0.028596  0.005105
# ipw                   0.011111  0.016634
# snipw                 0.010181  0.007922
# dr                    0.011152  0.015887
# sndr                  0.016778  0.025145
# switch-ipw (tau=1)    0.392239  0.003146
# switch-ipw (tau=100)  0.010062  0.008949
# switch-dr (tau=1)     0.032452  0.005018
# switch-dr (tau=100)   0.004864  0.004356
# dr-os (lambda=1)      0.030877  0.004588
# dr-os (lambda=100)    0.022244  0.004262
# ========================================
```

The above result can change with different situations.
You can try the evaluation of OPE with other experimental settings easily!
