# Example with Off-Policy Policy Learners


## Description

Here, we use synthetic bandit datasets to evaluate off-policy learners.
Specifically, we evaluate the performances of off-policy learners using the ground-truth policy value of an evaluation policy calculable with synthetic data.

## Evaluating Off-Policy Learners

In the following, we evaluate the performances of
- Random Policy (Random)
- Inverse Probability Weighting Policy Learner (IPWLearner)
- Policy Learner using Neural Networks (NNPolicyLearner)

See [our documentation](https://zr-obp.readthedocs.io/en/latest/_autosummary/obp.policy.offline.html) for the details about IPWLearner and NNPolicyLearner.

NNPolicyLearner can use the following OPE estimators as the objective function:
- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Doubly Robust with Optimistic Shrinkage (DRos)

See [our documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details about these estimators.

### Files
- [`./evaluate_off_policy_learners.py`](./evaluate_off_policy_learners.py) implements the evaluation of off-policy learners using synthetic bandit feedback data.
- [`./conf/hyperparams.yaml`](./conf/hyperparams.yaml) defines hyperparameters of some machine learning methods used to define regression model and IPWLearner.

### Scripts

```bash
# run evaluation of off-policy learners with synthetic bandit data
python evaluate_off_policy_learners.py\
    --n_rounds $n_rounds\
    --n_actions $n_actions\
    --dim_context $dim_context\
    --base_model_for_evaluation_policy $base_model_for_evaluation_policy\
    --base_model_for_reg_model $base_model_for_reg_model\
    --ope_estimator $ope_estimator\
    --n_hidden $n_hidden\
    --n_layers $n_layers\
    --activation $activation\
    --solver $solver\
    --batch_size $batch_size\
    --early_stopping\
    --random_state $random_state
```
- `$n_rounds` and `$n_actions` specify the number of rounds (or samples) and the number of actions of the synthetic bandit data.
- `$dim_context` specifies the dimension of context vectors.
- `$base_model_for_evaluation_policy` specifies the base ML model for defining evaluation policy and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$base_model_for_reg_model` specifies the base ML model for defining regression model and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$ope_estimator` specifies the OPE estimator for NNPolicyLearner and should be one of "dm", "ipw", "sipw", "dr", "sndr" or "dros".
- `$n_hidden` specifies the size of hidden layers in NNPolicyLearner.
- `$n_layers` specifies the number of hidden layers in NNPolicyLearner.
- `$activation` specifies theadam activation function for NNPolicyLearner and should be one of "identity", "tanh", "logistic", or "relu".
- `$solver` specifies the optimizer for NNPolicyLearner and should be one of "lbfgs", "sgd", or "adam".
- `$batch_size` specifies the batch size for NNPolicyLearner.
- `$early_stopping` enables early stopping of training of NNPolicyLearner.

For example, the following command compares the performances of the off-policy learners using the synthetic bandit feedback data with 100,00 rounds, 10 actions, five dimensional context vectors.

```bash
python evaluate_off_policy_learners.py\
    --n_rounds 10000\
    --n_actions 10\
    --dim_context 5\
    --base_model_for_evaluation_policy logistic_regression\
    --base_model_for_reg_model logistic_regression\
    --ope_estimator dr\
    --n_hidden 100\
    --n_layers 1\
    --activation relu\
    --solver adam\
    --batch_size 200\
    --early_stopping

# policy values of off-policy learners (higher means better)
# =============================================
# random_state=12345
# ---------------------------------------------
#                              policy value
# random_policy                    0.604339
# ipw_learner                      0.767615
# nn_policy_learner (with dr)      0.764302
# =============================================
```

The above result can change with different situations.
You can try the evaluation with other experimental settings easily.

