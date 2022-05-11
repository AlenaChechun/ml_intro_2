# -*- coding: utf-8 -*-
"""
Created on Wed May  4 23:56:31 2022

@author: alena.chechun
"""
from pathlib import Path

import click
import mlflow
import mlflow.sklearn

from .data import get_dataframe
from .pandas_profile import run_pandas


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default=Path("data/train.csv"),
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    "-out",
    "--output-path",
    default=Path("pandas_profiling.html"),
    show_default=True,
)
def pandas(
    dataset_path: Path,
    output_path: Path
) -> None:
    df = get_dataframe(dataset_path)
    with mlflow.start_run():
        run_pandas(df, output_path)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("output_path", output_path)
        mlflow.log_model('pandas_profiling')
        mlflow.end_run()
