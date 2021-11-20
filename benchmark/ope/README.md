# Benchmarking Off-Policy Evaluation

This directory includes the code to replicate the benchmark experiment done in the following paper.

Yuta Saito, Shunsuke Aihara, Megumi Matsutani, Yusuke Narita.<br>
**Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation**<br>
[https://arxiv.org/abs/2008.07146](https://arxiv.org/abs/2008.07146)


If you find this code useful in your research then please cite:
```
@article{saito2020open,
  title={Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation},
  author={Saito, Yuta and Shunsuke, Aihara and Megumi, Matsutani and Yusuke, Narita},
  journal={arXiv preprint arXiv:2008.07146},
  year={2020}
}
```

## Description
We use the (full size) open bandit dataset to evaluate and compare OPE estimators in a *realistic* and *reproducible* manner. Specifically, we evaluate the estimation performance of a wide variety of OPE estimators by comparing the policy values estimated by OPE with the on-policy policy value of an evaluation policy.

### Dataset
Please download the full [open bandit dataset](https://research.zozo.com/data.html) and put it in the `../open_bandit_dataset/` directory.

## Evaluating Off-Policy Estimators

In the benchmark experiment, we evaluate the estimation performance of the following OPE estimators.

- Direct Method (DM)
- Inverse Probability Weighting (IPW)
- Self-Normalized Inverse Probability Weighting (SNIPW)
- Doubly Robust (DR)
- Self-Normalized Doubly Robust (SNDR)
- Switch Doubly Robust (Switch-DR)
- Doubly Robust with Optimistic Shrinkage (DRos)

See Section 2 and Appendix B of [our paper](https://arxiv.org/abs/2008.07146) or the package [documentation](https://zr-obp.readthedocs.io/en/latest/estimators.html) for the details of these estimators.

For Switch-DR and DRos, we use a data-driven hyperparameter tuning method described in [Su et al.](https://arxiv.org/abs/1907.09623).
For estimators except for DM, we use the true action choice probability contained in Open Bandit Dataset.
For estimators except for IPW and SNIPW, we need to obtain a reward estimator.
We do this by using machine learning models (such as gradient boosting) implemented in *scikit-learn*.
We also use cross-fitting to avoid substantial bias from overfitting when obtaining a reward estimator.

## Requirements and Setup

The Python environment is built using [poetry](https://github.com/python-poetry/poetry). You can build the same environment as in our benchmark experiment by cloning the repository and running `poetry install` directly under the folder (if you have not install poetry yet, please run `pip install poetry` first.).

```bash
# clone the obp repository
git clone https://github.com/st-tech/zr-obp.git
cd benchmark/ope

# build the environment with poetry
poetry install

# run the benchmark experiment
poetry run python benchmark_off_policy_estimators.py ...
```

The versions of Python and used packages are as follows.

```
[tool.poetry.dependencies]
python = "^3.9,<3.10"
scikit-learn = "^0.24.2"
pandas = "^1.3.1"
numpy = "^1.21.1"
matplotlib = "^3.4.2"
hydra-core = "^1.0.7"
pingouin = "^0.4.0"
pyieoe = "^0.1.0"
obp = "^0.5.0"
```

## Files

- [benchmark_ope_estimators.py](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/benchmark_ope_estimators.py) implements the experimental workflow to evaluate and compare the above OPE estimators using Open Bandit Dataset. We summarize the detailed experimental protocol for evaluating OPE estimators using real-world data [here](https://zr-obp.readthedocs.io/en/latest/evaluation_ope.html).
- [benchmark_ope_estimators_hypara.py](https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/benchmark_ope_estimators.py) evaluates the effect of the hyperparameter choice on the OPE performance of DRos.
- [./conf/](./conf/) specifies experimental settings such as the number of random seeds.

## Scripts
The experimental workflow is implemented using [Hydra](https://github.com/facebookresearch/hydra). Below, we explain important experimental configurations.

```bash
# run evaluation of OPE experiments on the full open bandit dataset
poetry run python benchmark_ope_estimators.py\
    setting.n_seeds=$n_seeds\
    setting.campaign=$campaign\
    setting.behavior_policy=$behavior_policy\
    setting.sample_size=$sample_size\
    setting.reg_model=$reg_model\
    setting.is_timeseries_split=$is_time_series_split
```

- `$n_runs` specifies the number of random seeds used in the experiment.
- `$campaign` specifies the campaign considered in ZOZOTOWN and should be one of "all", "men", or "women".
- `$behavior_policy` specifies which policy in Random or Bernoulli TS (bts) is used as the behavior policy. This should be either of "random" or "bts".
- `$sample_size` specifies the number of samples contained in the logged bandit feedback used to conduct OPE.
- `$reg_model` specifies the base ML model for defining the regression model and should be one of "logistic_regression", "random_forest", or "lightgbm".
- `$is_timeseries_split` is whether the data is split based on timestamp or not. If true, the out-sample performance of OPE is tested. See the relevant paper for details.

Please see [`./conf/setting/default.yaml`](./conf/setting/default.yaml) for the default experimental configurations, which are to be used when they are not overridden.

It is possible to run multiple experimental settings easily by using the `--multirun (-m)` option of Hydra.
For example, the following script sweeps over all simulations including the three campaigns ('all', 'men',  and 'women') and two different behavior policies ('random' and 'bts').

```bash
poetry run python benchmark_ope_estimators.py setting.campaign=all,men,women setting.behavior_policy=random,bts --multirun
```

The experimental results (including the pairwise hypothesis test results) will be store in the `logs/` directory.
Our benchmark results and findings can be found in Section 5 of [our paper](https://arxiv.org/abs/2008.07146).
