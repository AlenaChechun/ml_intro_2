# -*- coding: utf-8 -*-
"""
Created on Thu May  5 18:57:41 2022

@author: alena.strapko
"""

import configparser

from pathlib import Path


class Config:
  
    def __init__(self):      
        self._PREPROCESS_SECTION = 'preprocess'
        self.path_ = None
        self.cfg = configparser.ConfigParser()
        self.drop_feature = None
    
    def read_config(self, path : Path):
        self.path_ = path
        self.cfg.read(path)

    def get_drop_features(self):
        if self._PREPROCESS_SECTION in self.cfg.sections():
            drop_features = self.cfg[self._PREPROCESS_SECTION]['drop'].split('\n')
            return drop_features
        raise Exception(f'[drop] list in [{self._PREPROCESS_SECTION}] section is not found in the file {self.path_}.')

    