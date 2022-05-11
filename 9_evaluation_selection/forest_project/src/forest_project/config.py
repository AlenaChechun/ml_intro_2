# -*- coding: utf-8 -*-
"""
Created on Thu May  5 18:57:41 2022

@author: alena.strapko
"""

import configparser

from typing import List
from pathlib import Path


class Config:

    def __init__(self) -> None:
        self._PREPROCESS = 'preprocess'
        self.cfg = configparser.ConfigParser()
        self.drop_feature = None

    def read_config(self, path: Path) -> None:
        self.path_ = path
        self.cfg.read(path)

    def get_drop_features(self) -> List[str]:
        if self._PREPROCESS in self.cfg.sections():
            drop_features = self.cfg[self._PREPROCESS]['drop'].split('\n')
            return drop_features
        raise Exception(f'{self._PREPROCESS} is not found in {self.path_}.')
