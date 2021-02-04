===============
About
===============
Motivated by the paucity of real-world data and implementation enabling the evaluation and comparison of OPE, we release the following open-source dataset and pipeline software for research uses.


Open Bandit Dataset (OBD)
------------------------------

*Open Bandit Dataset* is a public real-world logged bandit feedback data.
The dataset is provided by `ZOZO, Inc. <https://corp.zozo.com/en/about/profile/>`_, the largest Japanese fashion e-commerce company with over 5 billion USD market capitalization (as of May 2020).
The company uses multi-armed bandit algorithms to recommend fashion items to users in a large-scale fashion e-commerce platform called `ZOZOTOWN <https://zozo.jp/>`_.
The following figure presents examples of displayed fashion items as actions.

.. image:: ./_static/images/recommended_fashion_items.png
   :scale: 25%
   :align: center

We collected the data in a 7-day experiment in late November 2019 on three campaigns, corresponding to "all", "men's", and "women's" items, respectively.
Each campaign randomly uses either the Random policy or the Bernoulli Thompson Sampling (Bernoulli TS) policy for each coming user.
This dataset is unique in that it contains a set of multiple logged bandit feedback datasets collected by running two different data collection policies on the same platform.
This enables realistic and reproducible experimental comparisons of different OPE estimators for the first time.
These policies select three of the possible fashion items to each user.
Let :math:`\mathcal{I}` be a set of items and :math:`\mathcal{K}` be a set of recommendation positions.
The above figure shows that :math:`|\mathcal{K}|=3` for our data.
We assume that the reward (click indicator) depends only on the item and its position, which is a general assumption on the click generative model in the web industry :cite:`Li2018`.
Under the assumption, we can apply the standard OPE setup and estimators to our setting.
We describe some statistics of the dataset in the following.

.. image:: ./_static/images/statistics_of_obd.png
   :scale: 28%
   :align: center

The data is large and contains many millions of recommendation instances.
It also includes the probabilities that item :math:`a` is displayed at position :math:`k` by the data collection policies which are used to calculate the importance weight.
We share the full version of our data at https://research.zozo.com/data.html

Open Bandit Pipeline (OBP)
---------------------------------

*Open Bandit Pipeline* is an open-source Python software including a series of modules for implementing dataset preprocessing, policy learning methods, and OPE estimators.
Our software provides a complete, standardized experimental procedure for OPE research, ensuring that performance comparisons are fair, transparent, and reproducible.
It also enables fast and accurate OPE implementation through a single unified interface, simplifying the practical use of OPE.

.. image:: ./_static/images/overview.png
   :scale: 32%
   :align: center

Open Bandit Pipeline consists of the following main modules.

- **dataset module**: This module provides a data loader for Open Bandit Dataset and a flexible interface for handling logged bandit feedback. It also provides tools to generate synthetic bandit data and transform multi-class classification data to bandit data.
- **policy module**: This module provides interfaces for implementing new online and offline bandit policies. It also implements several standard policy learning methods.
- **simulator module**: This module provides functions for conducting offline bandit simulation. This module is necessary only when we want to implement the ReplayMethod to evaluate the performance of online or adaptive bandit policies with logged bandit data.
- **ope module**: This module provides interfaces for implementing OPE estimators. It also implements several standard and advanced OPE estimators.

Note that the pipeline provides flexible interfaces so that researchers can easily implement their own algorithms or estimators and evaluate them with our data and pipeline.
Moreover, the pipeline provides an interface for handling logged bandit feedback datasets.
Thus, practitioners can combine their own datasets with the pipeline and easily evaluate bandit algorithms' performances in their settings.

Please see `package reference <https://zr-obp.readthedocs.io/en/latest/obp.html>`_ for detailed information about Open Bandit Pipeline.

To our knowledge, our real-world dataset and pipeline are the first to include logged bandit datasets collected by running *multiple* different policies and codes to replicate the data collection policies.
These features enable the realistic and reproducible **evaluation of OPE** for the first time.
