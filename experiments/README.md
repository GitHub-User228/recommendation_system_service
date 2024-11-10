# Music Recommendation System

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-FDEE21?style=flat-square&logo=apachespark&logoColor=black)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![S3](https://img.shields.io/badge/S3-003366?style=for-the-badge)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-219ebc?style=for-the-badge)
![Pydantic](https://img.shields.io/badge/Pydantic-CC0066?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-yellow?style=for-the-badge)
![implicit](https://img.shields.io/badge/implicit-000000?style=for-the-badge&logo=implicit&logoColor=white)


## Description

This part of the repository contains scripts and files related to building the recommendation model.


## Project Structure

**[artifacts](/experiments/artifacts)**: This directory contains artifacts

**[config](/experiments/config)**: Configuration files directory
- [components.yaml](/experiments/config/components.yaml): Configuration for the project
- [logger_config.yaml](/experiments/config/logger_config.yaml): Configuration for the logger
- [spark_config.yaml](/experiments/config/spark_config.yaml): Configuration for the spark

**[mlflow_server](/experiments/mlflow_server)**: This directory contains scripts related to the MLflow server
- [start.sh](/experiments/mlflow_server/start.sh): Script to start the MLflow server
- [clean.sh](/experiments/mlflow_server/clean.sh): Script to remove deleted data from the MLflow server

**[notebooks](/experiments/notebooks)**: This directory jupyter notebooks where the experiments are performed
- [recommendations..ipynb](/experiments/notebooks/recommendations.ipynb): Notebook with the experiments

**[scripts](/experiments/scripts)**: This directory contains Python scripts used for conducting the experiments
- **[components](/experiments/scripts/components)**: A directory with the components of the project
- **[pipeline](/experiments/scripts/pipeline)**: A directory with the scripts to run the pipeline of components
    - [run.sh](/experiments/scripts/pipeline/run.sh): Script to run the pipeline of components. User can specify what components to run via the command line arguments

[.env_template](/experiments/.env_template): This is a template file for the environment variables

[requirements.txt](/experiments/requirements.txt): List of required Python packages

[setup.py](/experiments/setup.py): Setup file for packaging python scripts, so that all scripts can be easily accessable from any directory


## Getting Started

Follow the guides in [Instructions.md](Instructions.md) to check the installation process and how to run the pipeline.


## About the recommendation model

The recommendation model can be offline and online.

- `Offline model`: it is an ensemble model over a set of base recommenders with a ranking model on the top. The following base models were considered:
    - `ALS` (Alternating Least Squares) model (uses user-item interactions data)
    - `BPR` (Bayesian Personalized Ranking) model (uses user-item interactions data)
    - `Item2Item` model (uses item-features data)

    `Item2Item` model is essentially a hand-made model which works with item-features matrix. Given a user ID and user-item interactions data, it computes an average item-features vector and then finds top items according to the similarity criteria. To speed up the computation, the following was used:
    - `TruncatedSVD` algorithm - reduces the number of item features
    - `NearestNeighbors` algorithm - a much faster computation of the closest vectors for a given vector

    The ranking model is a `CatBoostClassifier`, which can easily work with missing data (e.g. some user-item candidates might have only one score from a base model) and can handle different scaling.

    Additionally, several user and item features were made - some of them significantly improved the quality of offline recommendations

- `Online model`: it is based on the items similarity. `ALS` and `BPR` models were used in order to calculate a set of similar items for each item that was observed when training each model. Hence, the online model can easily recommend similar items based on what tracks a user have recently listened to.

Additionally, the most popular tracks were retrieved so that they can be recommended for new users or in case it was not possible to give offline recommendations for a user (lack of relevant history data). Also those tracks were used for building the offline model (only new (item, user) pairs were added to preserve `Novelty`)


## Recommendation model metrics

The resulting recommendation model achived the following metrics:

- `Precision@10`: 1.1%
- `Recall@10`: 3.2%
- `NDCG@10`: 2.6%
- `CoverageItem@10`: 17.7%
- `CoverageUser@10`: 94.4%
- `Novelty@10`: 100%

---

Bucket: s3-student-mle-20240730-73c4e0c760