# Counterfactual Policy Search

## Description

## Running Counterfactual Policy Search

```
for model in lightgbm
do
    for context in 1
    do
        for camp in men
        do
            screen python run_cf_policy_search.py\
                --context_set $context\
                --base_model $model\
                --behavior_policy bts\
                --campaign $camp
        done
    done
done
```

```
python run_cf_policy_search.py --context_set 1 --base_model logistic_regression --campaign men --n_boot_samples 2 --test_size 0.9
```

## Results
