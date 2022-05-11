Homework for RS School Machine Learning course.
===============================================

This demo uses [**Forest Cover Type Prediction**](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

Usage
-----
This package allows you to train model.


1. Clone this repository to your machine.

```
git clone git@github.com:AlenaChechun/ml_intro_2.git
```

2. Switch to the branche '2022'

```
git checkout 2022
```

3. Download [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/train.csv* in repository's root).

4. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).

5. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):

```
poetry install --no-dev
```

6. Run *Pandas Profiling* with the following command:

```
poetry run pandas -d <path to csv with data> -out <path to save pandas profiling report in html>
```

7. Run train with the following command:

```
poetry run train -d <path to csv with data> -s <path to save trained model>
```

You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:

```
poetry run train --help
```

8. Run MLflow UI to see the information about experiments you conducted:

```
poetry run mlflow ui
```


Development
-----------

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

* Install all requirements (including dev requirements) to poetry environment:

```
poetry install
```

* Now you can use developer instruments, e.g. pytest:

```
poetry run pytest
```

* Run mypy

 mypy ``.\src\forest_project\``

* Run the pre-commit hooks that triggers mypy cheching for all files.

```
pre-commit run --all-files
```

Note that the pre-commit works only when the git repository was cloned and switched into '2022' branch.

```
git clone git@github.com:AlenaChechun/ml_intro_2.git
```

```
git checkout 2022
```

Assignment results
-------------------
#. [8] Track each experiment into MLFlow.

.. image:: img\8_mlflow.JPG

#. [12] Provide a screenshot that flake8 linting and formatting are passed.

.. image:: img\12_flake8.JPG

#. [13] MYPY result.

.. image:: img\13_mypy.JPG

#. [14] nox using.

.. image:: img\nox.JPG

#. Example of poetry env remove

.. image:: img\remove_env.JPG