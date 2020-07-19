============
About
============


Open Bandit Dataset (OBD)
------------------------------

*Open Bandit Dataset* is a public real-world logged bandit feedback data.
The dataset is provided by `ZOZO, Inc. <https://corp.zozo.com/en/about/profile/>`_, the largest Japanese fashion e-commerce company with over 5 billion USD market capitalization (as of May 2020).
The company uses multi-armed bandit algorithms to recommend fashion items to users in a large-scale fashion e-commerce platform called `ZOZOTOWN <https://zozo.jp/>`_.
The following figure presents examples of displayed fashion items as actions.

.. image:: ./_static/images/recommended_fashion_items.png
   :scale: 25%
   :align: center

We collected the data in a 7-day experiment in late November 2019 on three “campaigns,” corresponding to all, men's, and women's items, respectively.
Each campaign randomly uses either the Random algorithm or the Bernoulli Thompson Sampling (Bernoulli TS) algorithm for each user impression.
The following table describes the statistics of Open Bandit Dataset.

.. image:: ./_static/images/statistics_of_obd.png
   :scale: 25%
   :align: center


Open Bandit Pipeline (OBP)
---------------------------------

*Open Bandit Pipeline* is a series of implementations of dataset preprocessing, offline bandit simulation, and evaluation of OPE estimators.
This pipeline allows researchers to focus on building their bandit algorithm or OPE estimator and easily compare them with others’ methods in realistic and reproducible ways.
Thus, it facilitates reproducible research on bandit algorithms and off-policy evaluation.

.. image:: ./_static/images/overview.png
   :scale: 40%
   :align: center

Open Bandit Pipeline consists of the following main modules.

- **dataset module**: provides data loader for Open Bandit Dataset and a flexible interface for handling logged bandit feedback.
- **policy module**: provides interfaces for bandit algorithms and several standard algorithms
- **simulator module**: provides functions for conducting offline bandit simulation.
- **ope module**: provides interfaces for bandnit algorithms and several standard OPE estimators.


The pipeline also provides flexible interfaces for bandit algorithms and OPE estimators.
Therefore, researchers can easily implement their own methods and evaluate them with our data and pipeline.
In addition, the pipeline provides an interface for logged bandit feedback datasets.
Thus, practitioners can can combine their own datasets with the pipeline and easily evaluate the performance of bandit algorithms in their settings.

Please see `package reference <https://zr-obp.readthedocs.io/en/latest/obp.html>`_ for detailed information about Open Bandit Pipeline.
