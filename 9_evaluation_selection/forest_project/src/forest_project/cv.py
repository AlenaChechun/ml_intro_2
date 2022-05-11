# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:26:00 2022

@author: alena.chechun
"""
import click
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score

from .score import get_score


def kfolder(model: Pipeline,
            X: pd.DataFrame,
            y: pd.Series,
            n_splits: int,
            random_state: int,
            name_scorings: List[str],
) -> Tuple[List[float], List[float]]:
    score_mean = []
    score_std = []
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for scoring in name_scorings:
        score_arr = cross_val_score(model, X, y, scoring=scoring, cv=cv)
        score_mean.append(score_arr.mean())
        score_std.append(score_arr.std())
    return score_mean, score_std


def nested(model: Pipeline,
           model_params: List[Dict[str, object]],
           X: pd.DataFrame,
           y: pd.Series,
           n_splits: int,
           random_state: int,
           name_scorings: List[str],
           name_fit_core: str,
) -> Tuple[object, Dict[str, object], List[float]]:
    cv_out = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_score = 0
    best_cv = None
    best_score_list: List[float] = []
    for train_index, test_index in cv_out.split(X):
        X_train, X_test = X.iloc[train_index.tolist()], X.iloc[test_index.tolist()]
        y_train, y_test = y.iloc[train_index.tolist()], y.iloc[test_index.tolist()]
        click.echo(f'Splited train dataset: {X_train.shape}, {X_test.shape}')
        cv_inner = KFold(n_splits=(n_splits - 1), shuffle=True, random_state=random_state)
        cv = RandomizedSearchCV(model,
                          model_params,
                          scoring=name_scorings,
                          return_train_score=True,
                          cv=cv_inner,
                          refit=name_fit_core,
                          n_jobs=-1,
                          )
        cv = cv.fit(X_train, y_train)
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
        # print(cv.cv_results_)

        if cv.best_score_ > best_score:
            best_score_list = []
            best_score = cv.best_score_
            best_cv = cv
            best_params = cv.best_params_
            click.echo(f"nested-best_params_: {cv.best_params_}")
            click.echo(f"nested-train_score of {name_fit_core}: {cv.best_score_}")

            for name_score, idx in zip(name_scorings, range(len(name_scorings))):
                score = get_score(name_score, X_train, y_train, best_cv)
                click.echo(f"nested-train_score of {name_score}: {score}.")
                score = get_score(name_score, X_test, y_test, best_cv)
                best_score_list.append(score)
                click.echo(f"nested-test_score of {name_score}: {score}.")

    return best_cv, best_params, best_score_list
