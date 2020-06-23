============
About
============


Open Bandit Dataset (OBD)
------------------------------

*Open Bandit Dataset* is public real-world logged bandit feedback data.
The dataset is provided by `ZOZO, Inc. <https://corp.zozo.com/en/about/profile/>`_, the largest Japanese fashion e-commerce company with over 5 billion USD market capitalization (as of May 2020).
The company uses multi-armed bandit algorithms to recommend fashion items to users in a large-scale fashion e-commerce platform called `ZOZOTOWN <https://zozo.jp/>`_.
The following figure presents examples of displayed fashion items as actions.

.. image:: ./_static/images/recommended_fashion_items.png
   :scale: 25%
   :align: center

We collected the data in a 7-days experiment in late November 2019 on three “campaigns,” corresponding to all, men', and women' items, respectively.
Each campaign randomly uses either the Random algorithm or the Bernoulli Thompson Sampling (Bernoulli TS) algorithm for each user impression.

.. image:: ./_static/images/statistics_of_obd.png
   :scale: 25%
   :align: center


Open Bandit Pipeline (OBP)
---------------------------------

*Open Bandit Pipeline* is a series of implementations of dataset preprocessing, offline bandit simulation, and evaluation of OPE estimators.
This pipeline allows researchers to focus on building their OPE estimator and easily compare it with others’ methods in realistic and reproducible ways.
Thus, it facilitates reproducible research on bandit algorithms and off-policy evaluation.

.. image:: ./_static/images/overview.png
   :scale: 30%
   :align: center


