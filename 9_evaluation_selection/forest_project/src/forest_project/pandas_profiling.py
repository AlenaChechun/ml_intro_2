# -*- coding: utf-8 -*-
"""
Created on Wed May  4 23:34:13 2022

@author: alena.chechun
"""
import pandas as pd
from pandas_profiling import ProfileReport


def run_pandas(df : pd.DataFrame, result_path):
    prof = ProfileReport(df)
    prof.to_file(result_path)
