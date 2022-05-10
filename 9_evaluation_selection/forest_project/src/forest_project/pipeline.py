# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:56:45 2022

@author: alena.chechun
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest


def create_pipeline(
    model_obj: object,
    random_state: int,
    use_scaler: bool,
    use_variance: bool, variance : float,
    use_kbest: bool, k_best: int,
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
    pipeline_steps.append(
        (
            "cl", model_obj
        )
    )

    return Pipeline(steps=pipeline_steps)
