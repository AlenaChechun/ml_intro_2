# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:56:45 2022

@author: alena.chechun
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest


def create_pipeline(
    random_state: int,
    use_scaler: bool,
    use_variance: bool, variance : float,
    use_kbest: bool, k_best: int,
    use_logreg: bool, max_iter: int, logreg_C: float,
    use_rforest: bool, n_estimators: int, max_depth: int,
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    '''FEATURE ENGINEERING'''
    if use_variance:
        pipeline_steps.append(("select", VarianceThreshold(variance)))
    if use_kbest:
        pipeline_steps.append(("select", SelectKBest(f_classif, k=k_best)))
    if use_variance and use_kbest:
        raise Exception("Both feature selection approaches were selected. Please, use only one.")

    '''SELECT CLASSIFIER'''
    if use_logreg:
        pipeline_steps.append(
            (
                "cl",
                LogisticRegression(
                    random_state=random_state, max_iter=max_iter, C=logreg_C
                ),
            )
        )
    if use_rforest:
        pipeline_steps.append(
            (
                "cl",
                RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
                ),
            )
        )
    if use_variance ^ use_kbest:
        raise Exception("Both or no one Clasifier model was requred. Please, use --help.")

    return Pipeline(steps=pipeline_steps)
