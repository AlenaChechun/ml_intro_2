# -*- coding: utf-8 -*-
"""
Created on Wed May  4 23:34:13 2022

@author: alena.chechun
"""
from pathlib import Path
import pandas as pd
from pandas_profiling import ProfileReport


def run_pandas(df: pd.DataFrame,
               result_path: Path,
) -> None:
    prof = ProfileReport(df)
    prof.to_file(result_path)
