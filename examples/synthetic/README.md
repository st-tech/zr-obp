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
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)

For Switch-IPW, Switch-DR, and DRos, we try some different values of hyperparameters.
See [our documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details about these estimators.

### Files
- [`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators using synthetic bandit feedback data.
- [`./conf/hyperparams.yaml`](./conf/hyperparams.yaml) defines hyperparameters of some machine learning methods used to define regression model and IPWLearner.

### Scripts

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

# relative-ee of OPE estimators and their standard deviations (lower means accurate).
# It appears that the performances of some OPE estimators depend on the choice of their hyperparameters.
# =============================================
# random_state=12345
# ---------------------------------------------
#                          mean       std
# dm                   0.195878  0.012146
# ipw                  0.019335  0.013199
# snipw                0.007543  0.005196
# dr                   0.008099  0.006659
# sndr                 0.008054  0.004911
# switch-dr (tau=1)    0.195878  0.012146
# switch-dr (tau=100)  0.008099  0.006659
# dr-os (lambda=1)     0.195642  0.012151
# dr-os (lambda=100)   0.175285  0.012801
# =============================================
```

The above result can change with different situations.
You can try the evaluation of OPE with other experimental settings easily.
