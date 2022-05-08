# -*- coding: utf-8 -*-
"""
Created on Wed May  4 23:25:51 2022

@author: alena.chechun
"""
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataframe(csv_path: Path
) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def get_features(dataset : pd.DataFrame,
) -> pd.DataFrame:
    return dataset.drop("Cover_Type", axis=1)

def get_target(dataset : pd.DataFrame,
) -> pd.Series:
    return dataset["Cover_Type"]

def split_dataset(
    dataset : pd.DataFrame, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = get_features(dataset)
    target = get_target(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    return split_dataset(dataset, random_state, test_split_ratio)
