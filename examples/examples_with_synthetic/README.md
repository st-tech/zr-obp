# Examples with Synthetic Data


## Description

Here, we use synthetic bandit datasets and pipeline to evaluate OPE estimators.
Specifically, we evaluate the estimation performances of well-known off-policy estimators using the ground-truth policy value of an evaluation policy, which is calculable with synthetic data.

## Evaluating Off-Policy Estimators

In the following, we evaluate the estimation performances of
- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Switch Inverse Probability Weighting (Switch-IPW)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)

For Switch-IPW, Switch-DR, and DRos, we tried some different values of hyperparameters.

[`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators using synthetic bandit feedback datasets.
[`./conf/hyperparams.yaml`](./conf/hyperparams.yaml) defines hyperparameters of some machine learning methods used to define regression model and IPWLearner.

```bash
# run evaluation of OPE estimators with synthetic bandit data
python evaluate_off_policy_estimators.py\
    --n_runs $n_runs\
    --n_rounds $n_rounds\
    --n_actions $n_actions\
    --dim_context $dim_context\
    --base_model_for_evaluation_policy $base_model_for_evaluation_policy\
    --base_model_for_reg_model $base_model_for_reg_model\
    --n_jobs $n_jobs\
    --random_state $random_state
```
- `$n_runs` specifies the number of simulation runs in the experiment to estimate standard deviations of the performance of OPE estimators.
- `$n_rounds` and `$n_actions` specify the number of rounds (or samples) and the number of actions of the synthetic bandit data.
- `$dim_context` specifies the number of dimensions of context vectors.
- `$base_model_for_evaluation_policy` specifies the base ML model for defining evaluation policy and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$base_model_for_reg_model` specifies the base ML model for defining regression model and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$n_jobs` is the maximum number of concurrently running jobs.

For example, the following command compares the estimation performances of the OPE estimators using the synthetic bandit feedback data with 100,000 rounds, 30 actions, context vectors with five dimensions.

```bash
python evaluate_off_policy_estimators.py\
    --n_runs 20\
    --n_rounds 100000\
    --n_actions 30\
    --dim_context 5\
    --base_model_for_evaluation_policy logistic_regression\
    --base_model_for_reg_model logistic_regression\
    --n_jobs -1\
    --random_state 12345

# relative estimation errors of OPE estimators and their standard deviations (lower is better).
# our evaluation of OPE procedure suggests that DR and Switch-DR (tau=100) perform better than the other estimators.
# Moreover, it appears that the performances of some OPE estimators depend on the choice of hyperparameters.
# =============================================
# random_state=12345
# ---------------------------------------------
#                           mean       std
# dm                    0.029343  0.000410
# ipw                   0.002255  0.000587
# snipw                 0.001914  0.001268
# dr                    0.001645  0.000919
# sndr                  0.002550  0.000035
# switch-ipw (tau=1)    0.195059  0.000136
# switch-ipw (tau=100)  0.002255  0.000587
# switch-dr (tau=1)     0.046846  0.001251
# switch-dr (tau=100)   0.001645  0.000919
# dr-os (lambda=1)      0.028386  0.000369
# dr-os (lambda=100)    0.002516  0.001351
# =============================================
```

The above result can change with different situations.
You can try the evaluation of OPE with other experimental settings easily.
