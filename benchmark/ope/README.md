# Benchmarking Off-Policy Evaluation

## Description


## Training Regression Model

```
for model in logistic_regression
do
    for pi_b in random
    do
        for camp in men
        do
            python train_regression_model.py\
                --n_boot_samples 5\
                --base_model $model\
                --behavior_policy $pi_b\
                --campaign $camp
        done
    done
done
```


## Evaluating Off-Policy Estimators

```
for model in logistic_regression
do
    for pi_b in random
    do
        for camp in men
        do
            python benchmark_off_policy_estimators.py\
                --n_boot_samples 5\
                --base_model $model\
                --behavior_policy $pi_b\
                --campaign $camp
        done
    done
done
```

## Results
