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
from sklearn.metrics import accuracy_score

from .data import split_dataset, get_dataframe
from .config import Config as config
from .pipeline import create_pipeline
from .preprocess_data import preprocess

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-c",
    "--config-path",
    default="config.ini",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)


def train(
    dataset_path: Path,
    config_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    cfg = config()
    cfg.read_config(config_path)  
    dataset = get_dataframe(dataset_path)
    
    dataset = preprocess(
        dataset, 
        cfg.get_drop_features()
    )        
    X_train, X_test, y_train, y_test = split_dataset(
        dataset,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
        pipeline.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, pipeline.predict(X_test))
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy)
        click.echo(f"Accuracy: {accuracy}.")
        #dump(pipeline, save_model_path)
        #click.echo(f"Model is saved to {save_model_path}.")
        mlflow.end_run()