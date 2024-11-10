# Instructions on how to run the whole pipeline

Each instruction is executed from the repository directory

## 1. Installation

Install `openjdk-8-jdk` and `Conda` on the machine and initialize it before proceeding.

```python

# creating a conda environment
conda create -y --name venv_rss python=3.10.15

# activating the conda environment
conda activate venv_rss

# exporting the environment variables for the .env
conda env config vars set $(cat experiments/.env | tr '\n' ' ')

# reactivating the conda environment
conda deactivate
conda activate venv_rss

# cd to the directory with the experiments
cd experiments

# installing the dependencies
pip install -r requirements.txt
pip install -e .
```

## 2. Running the pipeline

Run the following command to run the pipeline:

```bash
bash experiments/scripts/pipeline/run.sh
```

Check [run.sh](/experiments/scripts/pipeline/run.sh) to see how to specify command line arguments so that only specific components are invoked

Also see [components.yaml](/experiments/config/components.yaml) and [settings.py](/experiments/scripts/settings.py) where you can modify the configuration of the components