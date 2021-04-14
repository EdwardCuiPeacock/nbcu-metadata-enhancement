# NBCU Metadata Enhancement


SETUP NEW MODEL:

STEP 0 - CHANGE DIRECTORY:

set working dir to training
```
cd training
```
STEP 1 - CHANGE CONFIGS:

- [training/pipeline_commands.sh](training/pipeline_commands.sh)
  - Make sure to change  `CUSTOM_TFX_IMAGE` and `PIPELINE_NAME`
  - `CUSTOM_TFX_IMAGE` must also match the image in [training/build.yaml](training/build.yaml) and `IMAGE` in [training/main/src/pipelines/configs.py](training/main/src/pipelines/configs.py)
  - `PIPELINE_NAME` must also match in [training/main/src/pipelines/configs.py](training/main/src/pipelines/configs.py)
- `GCS_BUCKET_NAME` needs to be changed in [training/main/src/pipelines/configs.py](training/main/src/pipelines/configs.py) to whatever bucket you want to save the model to
- pretty much everything high level is controlled in the config so change values to your hearts content i.e training length, input token length etc.
- MOST IMPORTANT MAYBE: make sure to change `MODEL_NAME` in [training/main/src/pipelines/configs.py](training/main/src/pipelines/configs.py) to the model name, (something descriptive) as this is what is used for eval and querying metrics for a model


STEP 2 - SOURCE COMMANDS:

```
source pipeline_commands.sh
```

STEP 3 - SETUP PIPELINE:

you can now just use the commands
```
build_pipeline
update_pipeline
run_pipeline
```
to build/update/launch your kubeflow pipeline

**NOTE: VERSIONS USED**
```
tfx=0.28
skaffold=v1.17.0 (should work with v2, just change the build.yaml)
tensorflow=2.4.1
```
-------------------

This codebase is split into two main folders:
- [training](training/README.md)
- [serving](./serving/README.md)

Each of these folders should have their own `readme`, which explain how to run the pipeline/service locally or in the cloud, and any setup instructions.

The purpose of this `readme` is for any information which is required for both training and serving folders. Add any information here that you feel is relevant to both training and serving.

----
## Python version management
We use [`pyenv`](https://github.com/pyenv/pyenv) to manage our Python version, and this is specified in a `.python-version` file in the `serving` and `training` directories.

To get started, `cd` into the `training` or `serving` directory, and make sure you have the correct python version installed with pyenv:

```
pyenv install `cat .python-version`
pyenv local `cat .python-version`  # Activate the correct python version
```
## Package management
We currently use [Poetry](https://python-poetry.org/) for python package management. We prefer to use Poetry rather than [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/) or similar as Poetry seems to be simpler and faster.

### Poetry

This is used to create a virtual environment and install all python packages inside. There are separate `pyproject.yaml` and `poetry.lock` files for both `training` and `serving` folders.

To install poetry just run:
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Once installed, you can install all the required packages (including dev packages) with the following:
```
poetry install
```
from either the `serving` dir or the `training` dir.

To enter the virtualenv in order to run commands with the installed packages, use
```
poetry shell
```
which will activate the virtualenv for you.

To add a new package, you can run:

```
poetry add <package>
```
See more advanced usage at https://python-poetry.org/docs/cli/#add.
Make sure to commit the new poetry.lock file to git if you add any new packages.

## Precommit

This project has an automatic linter setup which runs both [Black](https://github.com/psf/black) and [Flake8](https://flake8.pycqa.org/en/latest/). A good writeup of this solution is [here](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/).

To setup precommit:
```
# Install pre-commit
pip install pre-commit

# Setup pre-commit hooks
pre-commit install
```

To run the precommit on all files:
```
make pre-commit
```

In addition to being ran on every commit, this is ensured with the linter stage in `bibcd`. This builds a dockerfile and runs the pre-commit on all files.

## Building and Running the Pipeline 

You can use the [MLCLI](https://github.com/sky-uk/disco-mlops-mlcli/blob/master/README.md) tool in order to build and run your pipelines. 

Please look at the README for instructions on how to get started. 

