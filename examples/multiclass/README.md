# Example Experiment with Multi-class Classification Data


## Description

We use multi-class classification datasets to evaluate OPE estimators. Specifically, we evaluate the estimation performance of some well-known OPE estimators using the ground-truth policy value of an evaluation policy calculable with multi-class classification data.

## Evaluating Off-Policy Estimators

In the following, we evaluate the estimation performance of

- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)

For Switch-DR and DRos, we tune the built-in hyperparameters using SLOPE (Su et al., 2020;  Tucker et al., 2021), a data-driven hyperparameter tuning method for OPE estimators.
See [our documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details about these estimators.

### Files
- [`./evaluate_off_policy_estimators.py`](./evaluate_off_policy_estimators.py) implements the evaluation of OPE estimators using multi-class classification data.
- [`./conf/hyperparams.yaml`](./conf/hyperparams.yaml) defines hyperparameters of some ML methods used to define regression model.

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

For example, the following command compares the estimation performance (relative estimation error; relative-ee) of the OPE estimators using the digits dataset.

```bash
python evaluate_off_policy_estimators.py\
    --n_runs 30\
    --dataset_name digits\
    --eval_size 0.7\
    --base_model_for_behavior_policy logistic_regression\
    --alpha_b 0.4\
    --base_model_for_evaluation_policy random_forest\
    --alpha_e 0.9\
    --base_model_for_reg_model lightgbm\
    --n_jobs -1\
    --random_state 12345

# relative-ee of OPE estimators and their standard deviations (lower is better).
# =============================================
# random_state=12345
# ---------------------------------------------
#                mean       std
# dm         0.436541  0.017629
# ipw        0.030288  0.024506
# snipw      0.022764  0.017917
# dr         0.016156  0.012679
# sndr       0.022082  0.016865
# switch-dr  0.034657  0.018575
# dr-os      0.015868  0.012537
# =============================================
```

The above result can change with different situations. You can try the evaluation of OPE with other experimental settings easily.


## References

- Yi Su, Pavithra Srinath, Akshay Krishnamurthy. [Adaptive Estimator Selection for Off-Policy Evaluation](https://arxiv.org/abs/2002.07729), ICML2020.
- Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, Miroslav Dudík. [Doubly Robust Off-policy Evaluation with Shrinkage](https://arxiv.org/abs/1907.09623), ICML2020.
- George Tucker and Jonathan Lee. [Improved Estimator Selection for Off-Policy Evaluation](https://lyang36.github.io/icml2021_rltheory/camera_ready/79.pdf), Workshop on Reinforcement Learning
Theory at ICML2021.
- Yu-Xiang Wang, Alekh Agarwal, Miroslav Dudik. [Optimal and Adaptive Off-policy Evaluation in Contextual Bandits](https://arxiv.org/abs/1612.01205), ICML2017.
- Miroslav Dudik, John Langford, Lihong Li. [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601). ICML2011.
- Yuta Saito, Shunsuke Aihara, Megumi Matsutani, Yusuke Narita. [Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation](https://arxiv.org/abs/2008.07146). NeurIPS2021 Track on Datasets and Benchmarks.

