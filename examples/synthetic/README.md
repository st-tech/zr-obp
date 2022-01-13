# Example Experiment with Synthetic Bandit Data

## Description

We use synthetic bandit datasets to evaluate OPE estimators. Specifically, we evaluate the estimation performance of well-known  estimators using the ground-truth policy value of an evaluation policy calculable with synthetic data.

## Evaluating Off-Policy Estimators

In the following, we evaluate the estimation performance of

- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)

For Switch-DR and DRos, we tune the built-in hyperparameters using SLOPE, a data-driven hyperparameter tuning method for OPE estimators.
See [our documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details about these estimators.

### Files
- [`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators using synthetic bandit data.
- [`./conf/hyperparams.yaml`](./conf/hyperparams.yaml) defines hyperparameters of some ML methods used to define regression model and IPWLearner.

### Scripts

```bash
# run evaluation of OPE estimators with synthetic bandit data
python evaluate_off_policy_estimators.py\
    --n_runs $n_runs\
    --n_rounds $n_rounds\
    --n_actions $n_actions\
    --dim_context $dim_context\
    --beta $beta\
    --base_model_for_evaluation_policy $base_model_for_evaluation_policy\
    --base_model_for_reg_model $base_model_for_reg_model\
    --n_jobs $n_jobs\
    --random_state $random_state
```
- `$n_runs` specifies the number of simulation runs in the experiment to estimate standard deviations of the performance of OPE estimators.
- `$n_rounds` and `$n_actions` specify the sample size and the number of actions of the synthetic bandit data, respectively.
- `$dim_context` specifies the dimension of context vectors.
- `$beta` specifies the inverse temperature parameter to control the behavior policy.
- `$base_model_for_evaluation_policy` specifies the base ML model for defining evaluation policy and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$base_model_for_reg_model` specifies the base ML model for defining regression model and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$n_jobs` is the maximum number of concurrently running jobs.

For example, the following command compares the estimation performance (relative estimation error; relative-ee) of the OPE estimators using synthetic bandit data with 10,000 samples, 30 actions, five dimensional context vectors.

```bash
python evaluate_off_policy_estimators.py\
    --n_runs 20\
    --n_rounds 10000\
    --n_actions 30\
    --dim_context 5\
    --beta -3\
    --base_model_for_evaluation_policy logistic_regression\
    --base_model_for_reg_model logistic_regression\
    --n_jobs -1\
    --random_state 12345

# relative-ee of OPE estimators and their standard deviations (lower means accurate).
# =============================================
# random_state=12345
# ---------------------------------------------
#                mean       std
# dm         0.074390  0.024525
# ipw        0.009481  0.006899
# snipw      0.006665  0.004541
# dr         0.006175  0.004245
# sndr       0.006118  0.003997
# switch-dr  0.006175  0.004245
# dr-os      0.021951  0.013337
# =============================================
```

The above result can change with different situations.
You can try the evaluation of OPE with other experimental settings easily.
