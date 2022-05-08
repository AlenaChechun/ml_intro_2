# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:26:00 2022

@author: alena.chechun
"""
import pandas as pd
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def kfolder(model: Pipeline,
            X: pd.DataFrame,
            y: pd.Series,
            n_splits: int,
            random_state: int,
            name_scorings: list,
) -> Tuple[list, list]:
    score_mean = []
    score_std = []
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for scoring in name_scorings:
        score_arr = cross_val_score(model, X, y, scoring=scoring, cv=cv)
        score_mean.append(score_arr.mean())
        score_std.append(score_arr.std())
    return score_mean, score_std
