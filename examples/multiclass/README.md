# Example with Multi-class Classification Data


## Description

Here, we use multi-class classification datasets to evaluate OPE estimators.
Specifically, we evaluate the estimation performances of well-known off-policy estimators using the ground-truth policy value of an evaluation policy calculable with multi-class classification data.

## Evaluating Off-Policy Estimators

In the following, we evaluate the estimation performances of
- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)

For Switch-DR and DRos, we try some different values of hyperparameters.
See [our documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details about these estimators.

### Files
- [`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators using multi-class classification data.
- [`./conf/hyperparams.yaml`](./conf/hyperparams.yaml) defines hyperparameters of some machine learning methods used to define regression model.

### Scripts

```bash
# run evaluation of OPE estimators with multi-class classification data
python evaluate_off_policy_estimators.py\
    --n_runs $n_runs\
    --dataset_name $dataset_name \
    --eval_size $eval_size \
    --base_model_for_behavior_policy $base_model_for_behavior_policy\
    --alpha_b $alpha_b \
    --base_model_for_evaluation_policy $base_model_for_evaluation_policy\
    --alpha_e $alpha_e \
    --base_model_for_reg_model $base_model_for_reg_model\
    --n_jobs $n_jobs\
    --random_state $random_state
```
- `$n_runs` specifies the number of simulation runs in the experiment to estimate standard deviations of the performance of OPE estimators.
- `$dataset_name` specifies the name of the multi-class classification dataset and should be one of "breast_cancer", "digits", "iris", or "wine".
- `$eval_size` specifies the proportion of the dataset to include in the evaluation split.
- `$base_model_for_behavior_policy` specifies the base ML model for defining behavior policy and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$alpha_b`: specifies the ratio of a uniform random policy when constructing a behavior policy.
- `$base_model_for_evaluation_policy` specifies the base ML model for defining evaluation policy and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$alpha_e`: specifies the ratio of a uniform random policy when constructing an evaluation policy.
- `$base_model_for_reg_model` specifies the base ML model for defining regression model and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$n_jobs` is the maximum number of concurrently running jobs.

For example, the following command compares the estimation performances (relative estimation error; relative-ee) of the OPE estimators using the digits dataset.

```bash
python evaluate_off_policy_estimators.py\
    --n_runs 20\
    --dataset_name digits\
    --eval_size 0.7\
    --base_model_for_behavior_policy logistic_regression\
    --alpha_b 0.8\
    --base_model_for_evaluation_policy logistic_regression\
    --alpha_e 0.9\
    --base_model_for_reg_model logistic_regression\
    --n_jobs -1\
    --random_state 12345

# relative-ee of OPE estimators and their standard deviations (lower is better).
# It appears that the performances of some OPE estimators depend on the choice of their hyperparameters.
# =============================================
# random_state=12345
# ---------------------------------------------
#                           mean       std
# dm                    0.093439  0.015391
# ipw                   0.013286  0.008496
# snipw                 0.006797  0.004094
# dr                    0.007780  0.004492
# sndr                  0.007210  0.004089
# switch-dr (tau=1)     0.173282  0.020025
# switch-dr (tau=100)   0.007780  0.004492
# dr-os (lambda=1)      0.079629  0.014008
# dr-os (lambda=100)    0.008031  0.004634
# =============================================
```

The above result can change with different situations.
You can try the evaluation of OPE with other experimental settings easily.
