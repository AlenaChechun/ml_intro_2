# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:45:01 2022

@author: alena.chechun
"""

from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn

from .data import split_dataset, get_dataframe, get_features, get_target
from .config import Config as config
from .pipeline import create_pipeline
from .preprocess_data import preprocess
from .cv import kfolder, nested
from .score import get_score


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    "-c",
    "--config-path",
    default="config.ini",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    "-c",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True),
    show_default=True,
)
@click.option(
    "--random-state",
    default=23,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-variance",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--variance",
    default=0.8,
    type=click.FloatRange(0, 1),
    show_default=True,
)
@click.option(
    "--use-kbest",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--k-best",
    default=30,
    type=int,
    show_default=True,
)
@click.option(
    "--model",
    default='RFOREST',
    type=click.Choice(['LOGISTIC', 'RFOREST'], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--max-iter",
    default=1000,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    show_default=True,
    help='[default: None]'
)
@click.option(
    "--cv-kfolder",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--cv-nested",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--n-splits",
    default=5,
    type=int,
    show_default=True,
)


def train(
    dataset_path: Path,
    config_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    use_variance: bool, variance: int,
    use_kbest: bool, k_best: int,
    model: str,
    max_iter: int, logreg_c: float,    # params for logistic regression
    n_estimators:int, max_depth: int,  # params for random
    cv_kfolder: bool, cv_nested: bool, n_splits: int,
) -> None:
    cfg = config()
    cfg.read_config(config_path)
    scorings = ['accuracy', 'f1_weighted', 'roc_auc_ovr']   # https://scikit-learn.org/stable/modules/model_evaluation.html
    dataset = get_dataframe(dataset_path)

    dataset = preprocess(
        dataset,
        cfg.get_drop_features()
    )

    if model == 'LOGISTIC':
        use_logreg = True
        use_rforest = False
        model_params = [{'cl__max_iter': [],
                         'cl__C': [0.01, 0.01, 1, 10, 100]
                        }]
    elif model == 'RFOREST':
        use_logreg = False
        use_rforest = True
        model_params = [{'cl__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                         'cl__criterion': ['entropy', 'gini'],
                         'cl__max_depth': list(range(10, 20))
                         }]

    with mlflow.start_run():
        pipeline = create_pipeline(
            random_state,
            use_scaler,
            use_variance, variance,
            use_kbest, k_best,
            use_logreg, max_iter, logreg_c,
            use_rforest, n_estimators, max_depth
        )

        if cv_nested:
            nested(
                pipeline,
                model_params,
                get_features(dataset),
                get_target(dataset),
                n_splits,
                random_state,
                scorings
            )

        elif cv_kfolder:
            score_mean, score_std = kfolder(
                pipeline,
                get_features(dataset),
                get_target(dataset),
                n_splits,
                random_state,
                scorings
            )
        else:
            X_train, X_test, y_train, y_test = split_dataset(
                dataset,
                random_state,
                test_split_ratio,
            )
            pipeline.fit(X_train, y_train)

            score_mean = []
            score_std = []
            for name_score in scorings:
                score_mean.append(get_score(name_score, X_test, y_test, pipeline))
                score_std.append(0)


        mlflow.sklearn.log_model(pipeline, model)       # dump pickle module
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_variance", use_variance)
        if use_variance:
            mlflow.log_param("variance", variance)
        mlflow.log_param("use_kbest", use_kbest)
        if use_kbest:
            mlflow.log_param("k_best", k_best)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_param("output_path", save_model_path)

        for name_score, idx in zip(scorings, range(len(scorings))):
            mlflow.log_metric(name_score, score_mean[idx])
            mlflow.log_metric(name_score + 'std', score_std[idx])
            click.echo(f"{name_score}: {score_mean[idx]} +/- {score_std[idx]}.")

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
        mlflow.end_run()
