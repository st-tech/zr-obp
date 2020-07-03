```bash
python evaluate_counterfactual_policy.py\
    --context_set 1\
    --counterfactual_policy logistic_ucb\
    --epsilon 0.1\
    --behavior_policy bts\
    --campaign men
```

```bash
python evaluate_off_policy_estimators.py\
    --n_splits 5\
    --counterfactual_policy random\
    --behavior_policy bts\
    --campaign men
```
