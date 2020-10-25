# Benchmarking Off-Policy Evaluation

## Description
We use the (full size) open bandit dataset to evaluate OPE estimators in a *realistic* and *reproducible* manner. Specifically, we evaluate the estimation performances of a wide variety of existing off-policy estimators by comparing the estimated policy values with the ground-truth policy value of an evaluation policy contained in the data.

### Dataset
Please download the full [open bandit dataset](https://research.zozo.com/data.html) and put it as the `../open_bandit_dataset/` directory.

### Configurations

- [`conf/hyperparams.yaml`](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/conf/hyperparams.yaml)
  The hyperparameters of the some ML model that are used as the regression model in model dependent OPE estimators such as DM and DR.

## Training Regression Model

Here we train a regression model using some machine learning method. This will be used in the model-dependent estimators such as DM or DR.
[train_regression_model.py](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/train_regression_model.py) implements the training process of the regression model.

```
python train_regression_model.py\
    --n_runs $n_runs\
    --base_model $base_model\ "logistic_regression" or "lightgbm"
    --behavior_policy $behavior_policy\ "random" or "bts"
    --campaign $campaign\ # "men", "women", or "all"
    --is_timeseries_split $is_timeseries_split\ # in-sample or out-sample
    --n_sim_to_compute_action_dist $n_sim_to_compute_action_dist\
    --is_mrdr $is_mrdr\ # use "more robust doubly robust" option or not
    --random_state $random_state
```

where
- `$n_runs` specifies the number of simulation runs with different bootstrap samples in the experiment.
- `$base_model` specifies the base ML model for defining the regression model and should be one of "logistic_regression" or "lightgbm".
- `$campaign` specifies the campaign and should be one of 'all', 'men', or 'women'.
- `$n_sim_to_compute_action_dist` is the number of monte carlo simulation to compute the action distribution of a given evaluation policy.
- `$test_size` specifies the proportion of the dataset to include in the test split (when `$is_timeseries_split` is applied.)
- `$is_timeseries_split` is whether the data is split based on timestamp or not. If true, the out-sample performance of OPE is tested. See the relevant paper for details.
- `$is_mrdr` is whether the regression model is trained by the more robust doubly robust way or not. See the relevant paper for details.

For example, the following command trains the regression model based on logistic regression on the logged bandit feedback data collected by the Random policy (as behavior policy) in "All" campaign by the more robust doubly robust way.

```bash
python train_regression_model.py\
    --n_runs 10\
    --base_model logistic_regression\
    --behavior_policy random\
    --campaign all\
    --is_mrdr True\
    --is_timeseries_split False
```


```
for model in logistic_regression
do
    for pi_b in bts
    do
        for camp in all
        do
            for is_mrdr in True False
            do
                for is_timeseries in True False
                do
                    screen python train_regression_model.py\
                        --n_runs 10\
                        --base_model $model\
                        --behavior_policy $pi_b\
                        --campaign $camp\
                        --is_mrdr $is_mrdr\
                        --is_timeseries_split $is_timeseries
                done
            done
        done
    done
done
```


## Evaluating Off-Policy Estimators

Here, we evaluate the estimation performances of the following OPE estimators:

- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Switch Inverse Probability Weighting (Switch-IPW)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)
-  More Robust Doubly Robust (MRDR)

For Switch-IPW, Switch-DR, and DRos, we use some different values of hyperparameters.

[benchmark_off_policy_estimators.py](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/benchmark_off_policy_estimators.py) implements the evaluation and comparison of OPE estimators using the open bandit dataset.
Note that you have to finish training a regression model (see the above section) before conducting the evaluation of OPE in the corresponding setting.
The detailed experimental procedures and results can be found in Section 5 of https://arxiv.org/abs/2008.07146.

```
# run evaluation of OPE estimators with the full open bandit data
python benchmark_off_policy_estimators.py\
    --n_runs $n_runs\
    --base_model $base_model\ "logistic_regression" or "lightgbm"
    --behavior_policy $behavior_policy\ "random" or "bts"
    --campaign $campaign\ # "men", "women", or "all"
    --n_sim_to_compute_action_dist $n_sim_to_compute_action_dist\
    --test_size $test_size\
    --is_timeseries_split\ # in-sample or out-sample
    --random_state $random_state
```
where
- `$n_runs` specifies the number of simulation runs with different bootstrap samples in the experiment to estimate standard deviations of the performance of OPE estimators (i.e., relative estimation error).
- $base_model_for_evaluation_policy specifies the base ML model for defining the regression model and should be one of "logistic_regression" or "lightgbm".
- `$campaign` specifies the campaign and should be one of 'all', 'men', or 'women'.
- `$n_sim_to_compute_action_dist` is the number of monte carlo simulation to compute the action distribution of a given evaluation policy.
- `$test_size` specifies the proportion of the dataset to include in the test split (when `$is_timeseries_split` is applied.)
- `$is_timeseries_split` is whether the data is split based on timestamp or not. If true, the out-sample performance of OPE is tested. See the relevant paper for details.

For example, the following command compares the estimation performances of the listed OPE estimators by using Bernoulli TS as evaluation policy and Random as behavior policy in "All" campaign in the out-sample situation.

```bash
python benchmark_off_policy_estimators.py\
    --n_runs 3\
    --base_model logistic_regression\
    --behavior_policy random\
    --campaign all\
    --test_size 0.3\
    --is_timeseries_split True
```


```
for model in logistic_regression
do
    for pi_b in bts
    do
        for camp in all
        do
            for is_timeseries in True False
            do
                screen python benchmark_off_policy_estimators.py\
                    --n_runs 10\
                    --base_model $model\
                    --behavior_policy $pi_b\
                    --campaign $camp\
                    --is_timeseries_split $is_timeseries
            done
        done
    done
done
```

## Results

Here, we report the results of the benchmarking experiments on OPE estimators.
