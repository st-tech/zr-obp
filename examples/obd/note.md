
# regression model
```bash
python eval_regression_model.py\
    --n_splits 1\
    --behavior_policy bts\
    --campaign men
```

```bash
python off_policy_estimator_selection.py\
    --n_splits 1\
    --n_estimators 10\
    --behavior_policy bts\
    --campaign men
```

```bash
python cf_policy_selection.py\
    --n_splits 1\
    --n_estimators 10\
    --context_set 1\
    --counterfactual_policy logistic_ucb\
    --epsilon 0.1\
    --behavior_policy bts\
    --campaign men
```
