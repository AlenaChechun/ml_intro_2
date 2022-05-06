# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:56:45 2022

@author: alena.chechun
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest


def create_pipeline(
    use_scaler: bool,
    use_variance: bool, variance : float,
    use_kbest: bool, k_best: int,
    max_iter: int, logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    if use_variance:
        pipeline_steps.append(("select", VarianceThreshold(variance)))
    if use_kbest:
        pipeline_steps.append(("select", SelectKBest(f_classif, k=k_best)))
    if use_variance and use_kbest:
        raise Exception("Both feature selection approaches were selected. Please, use only one.")
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
