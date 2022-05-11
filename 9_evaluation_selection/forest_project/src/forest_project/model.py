# -*- coding: utf-8 -*-
"""
Created on Tue May 10 18:51:37 2022

@author: alena.chechun
"""
from typing import Dict, List
import mlflow
import click

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


MODEL_LOGISTIC = 'LOGISTIC'
MODEL_LOGISTIC_PARAMS = ['max_iter', 'C']
MODEL_LOGISTIC_CV_PARAMS = [{'cl__max_iter': [100, 1000, 10000],
                            'cl__C': [0.01, 0.01, 1, 10, 100]}]

MODEL_RFOREST = 'RFOREST'
MODEL_RFOREST_PARAMS = ['n_estimators', 'max_depth', 'criterion']
MODEL_RFOREST_CV_PARAMS = [{'cl__n_estimators':
                            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                            'cl__criterion': ['entropy', 'gini'],
                            'cl__max_depth': list(range(10, 20))}]


class Model:

    def __init__(self,
                 name: str,
                 random_state: int,
                 **kwargs: float,
                 ) -> None:
        self.params = {}
        self.name = name
        if self.name == MODEL_LOGISTIC:
            self.params_name = MODEL_LOGISTIC_PARAMS
        elif self.name == MODEL_RFOREST:
            self.params_name = MODEL_RFOREST_PARAMS
        else:
            raise Exception(f"Unknown Model '{self.model}")

        for key, value in kwargs.items():
            if key in self.params_name:
                self.params[key] = value
            elif key in MODEL_LOGISTIC_PARAMS + MODEL_RFOREST_PARAMS:
                print(f'unused param is {key}')
            else:
                raise Exception(f"Parameter is '{key} for {self.name}.")
        self.setup_model(random_state)

    def setup_model(self, random_state: int) -> None:
        if self.name == MODEL_LOGISTIC:
            self.model = LogisticRegression(
                random_state=random_state,
                max_iter=self.params['max_iter'],
                C=self.params['C']
                )
        elif self.name == MODEL_RFOREST:
            self.model = RandomForestClassifier(
                random_state=random_state,
                n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth']
                )
        else:
            raise Exception(f"Unknown Model '{self.model}")

    def get_model(self) -> object:
        return self.model

    def get_cv_params(self) -> List[Dict[str, object]]:
        if self.name == MODEL_LOGISTIC:
            return MODEL_LOGISTIC_CV_PARAMS
        elif self.name == MODEL_RFOREST:
            return MODEL_RFOREST_CV_PARAMS
        else:
            raise Exception(f"Unknown Model '{self.model}")

    def set_params(self, params: Dict[str, object]) -> None:
        for key, value in params.items():
            found_param = False
            for param in self.params_name:
                if key.endswith(param):
                    self.params[param] = value
                    found_param = True
                    msg = f"Update the model {self.name} : {param} = {value}."
                    click.echo(msg)
                    break
            if found_param is False:
                msg = f"Model Param '{key}' is updating {self.name}"
                raise Exception(msg)
        pass

    def mlflow_log_param(self, mlflow: mlflow) -> None:
        for key, value in self.params.items():
            mlflow.log_param(key, value)
        pass
