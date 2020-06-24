# Examples with OBD
Here, we use the dataset and pipeline to implement and evaluate OPE.
We first evaluate well-known off-policy estimators with the ground-truth performance of a counterfactual policy.
We then select and use such an estimator to improve the platform’s fashion item recommendation.

## Additional Implementations

- [`dataset.py`](./dataset.py):
    We implement two ways of engeering original features.
    This includes **CONTEXT SET 1** (user features only, such as age, gender, and length of  membership) and **CONTEXT SET 2** (Context Set 1 plus user-item affinity induced by the number of past clicks observed between each user-item pair).

## Running experiments

**Step 1. Evaluating a regression model**

We evaluate LightGBM as a regression model to be used in DM and DR.

```bash
python eval_regression_model.py\
    --n_splits 1\ # the number of train-test splits
    --behavior_policy bts\ # a behavior policy
    --campaign men # a campaign

# click prediction performance of a regression model
# =========================
# AUC: 0.615171
# RCE: 0.010273
# =========================
```

**Step 2. Selecting Off-policy Estimator** (Section 5.1)

We select the best off-policy estimator among DM, IPW, and DR.

```bash
python off_policy_estimator_selection.py\
    --n_splits 1\
    --n_estimators 10\ # the number of bagged samples used in bagging aggregation
    --behavior_policy bts\
    --campaign men

# relative estiamtion erros of three OPE estimators
# =========================
# random_state=0
# -----
# DM: 0.342532
# IPW: 0.684355
# DR: 1.035945
# =========================
```


**Step 3. Selecting Counterfactual Policy** (Section 5.2)

We evaluate the performance of counterfactual policies based on logistic contextual bandit with OPE estimators.

```bash
python cf_policy_selection.py\
    --n_splits 1\
    --n_estimators 10\
    --context_set 1\ # a used context set
    --counterfactual_policy logistic_ucb\ # a used counterfactual bandit algorithm
    --epsilon 0.1\ # an exploration hyperparameter
    --behavior_policy bts\
    --campaign men

# estimated policy values relative to the behavior policy (the Bernoulli TS here) of a counterfactual policy (the Logistic UCB with Context Set 1 here) by three OPE estimators
# =========================
# random_state=0
# -----
# DM: 0.826714
# IPW: 0.922185
# DR: 1.013117
# =========================
```

