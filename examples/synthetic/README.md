# Example with Synthetic Bandit Data


## Description

Here, we use synthetic bandit datasets to evaluate OPE estimators.
Specifically, we evaluate the estimation performances of well-known off-policy estimators using the ground-truth policy value of an evaluation policy calculable with synthetic data.

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

For Switch-IPW, Switch-DR, and DRos, we try some different values of hyperparameters.
See [our documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details about these estimators.

[`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators using synthetic bandit feedback data.
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
- `$dim_context` specifies the dimension of context vectors.
- `$base_model_for_evaluation_policy` specifies the base ML model for defining evaluation policy and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$base_model_for_reg_model` specifies the base ML model for defining regression model and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$n_jobs` is the maximum number of concurrently running jobs.

For example, the following command compares the estimation performances (relative estimation error; relative-ee) of the OPE estimators using the synthetic bandit feedback data with 100,000 rounds, 30 actions, five dimensional context vectors.

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

# relative-ee of OPE estimators and their standard deviations (lower is better).
# It appears that the performances of some OPE estimators depend on the choice of their hyperparameters.
# =============================================
# random_state=12345
# ---------------------------------------------
#                           mean       std
# dm                    0.011110  0.000565
# ipw                   0.001953  0.000387
# snipw                 0.002036  0.000835
# dr                    0.001573  0.000631
# sndr                  0.001578  0.000625
# switch-ipw (tau=1)    0.138523  0.000514
# switch-ipw (tau=100)  0.001953  0.000387
# switch-dr (tau=1)     0.021875  0.000414
# switch-dr (tau=100)   0.001573  0.000631
# dr-os (lambda=1)      0.010952  0.000567
# dr-os (lambda=100)    0.001835  0.000884
# =============================================
```

The above result can change with different situations.
You can try the evaluation of OPE with other experimental settings easily.
