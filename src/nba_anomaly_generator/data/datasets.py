import pandas as pd


def load_lal(**kwargs):
    default_kwargs = dict(index_col = 0)
    kwargs = dict(**default_kwargs, **kwargs)
    return pd.read_csv("lal.csv", **kwargs)
