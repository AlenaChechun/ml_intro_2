# -*- coding: utf-8 -*-
"""
Created on Sun May  8 22:29:27 2022

@author: alena.chechun
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# https://scikit-learn.org/stable/modules/model_evaluation.html
def get_score(score_name: str,
              X_test: pd.DataFrame,
              y_test: pd.Series,
              pipeline: Pipeline,
              ) -> float:
    score = 0
    y_pred = pipeline.predict(X_test)
    if score_name == 'accuracy':
        score = accuracy_score(y_test,
                               y_pred
                               )
    elif score_name == 'f1_weighted':
        score = f1_score(y_test,
                         y_pred,
                         average='weighted'
                         )
    elif score_name == 'f1_micro':
        score = f1_score(y_test,
                         y_pred,
                         average='micro',
                         )
    elif score_name == 'roc_auc_ovr':
        score = roc_auc_score(y_test,
                              pipeline.predict_proba(X_test),
                              multi_class='ovr'
                              )
    else:
        msg = f"Scoring Classification _{score_name}_ is not found."
        raise Exception(msg)
    return score
