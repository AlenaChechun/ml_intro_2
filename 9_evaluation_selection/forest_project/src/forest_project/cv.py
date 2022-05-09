# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:26:00 2022

@author: alena.chechun
"""
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

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
) -> None:
    cv_out = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, test_index in cv_out.split(X):
        X_train, X_test = X.iloc[train_index.tolist()], X.iloc[test_index.tolist()]
        y_train, y_test = y.iloc[train_index.tolist()], y.iloc[test_index.tolist()]
        print(X_train.shape, X_test.shape)
        cv_inner = KFold(n_splits=(n_splits - 1), shuffle=True, random_state=random_state)
        cv = GridSearchCV(model,
                          model_params,
                          scoring=name_scorings,
                          return_train_score=True,
                          cv=cv_inner,
                          refit=False,
                          )
        result = cv.fit(X_train, y_train)
        best_model = result.best_estimator_
        print(result.best_params_)
        #print(best_model)
        print(f"nested-train_score: {result.best_score_}")

        for name_score, idx in zip(name_scorings, range(len(name_scorings))):
            score = get_score(name_score, X_test, y_test, best_model)
            print(f"nested-test_score: {score}.")
