# Instructions on how to run the pipeline

Each instruction is executed from the repository directory

## 1. Installation

Fill in the `.env_template` file with the necessary environment variables and rename it to `.env`.


Install `openjdk-8-jdk`

```
sudo apt-get install openjdk-8-jdk
```

Install `Conda` on the machine and initialize it.

Now, follow the instructions below to create the conda environment and install the dependencies.

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

## 2. Start the MLFlow server

Run the following command to start the MLFlow server:

```bash
bash experiments/mlflow_server/start.sh
```

## 3. Running the pipeline

Run the following command to run the pipeline:

```bash
bash experiments/scripts/pipeline/run.sh -p true -m true -f 1,2,3 -a 1,2,3,4,5,6 -b 1,2,3,4,5,6 -i 1,2,3,4,5,6 -t 1,2,3,4,5,6 -e 1,2,3,4,5
```

Check [run.sh](/experiments/scripts/pipeline/run.sh) to see how to specify command line arguments so that only specific components are invoked

Also see [components.yaml](/experiments/config/components.yaml) and [settings.py](/experiments/scripts/settings.py) where you can modify the configuration of the components