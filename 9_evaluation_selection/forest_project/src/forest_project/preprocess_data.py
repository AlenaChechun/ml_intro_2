# -*- coding: utf-8 -*-
"""
Created on Thu May  5 01:24:58 2022

@author: alena.chechun
"""
import pandas as pd
import click


def preprocess(df: pd.DataFrame,
               drop_columns: list,
) -> pd.DataFrame:
    for column in drop_columns:
        df = drop_column(df, column)
    return df

def drop_column(df: pd.DataFrame,
                column_name: str,
 ) -> pd.DataFrame:
    click.echo(f'Dtop the feature {column_name}.')
    df = df.drop(column_name, axis='columns')
    return df
