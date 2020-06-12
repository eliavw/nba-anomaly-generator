import numpy as np
import pandas as pd


# Prepare dataframe
def assert_and_rng_and_prep_df(df, rng=None, random_state=42):
    """These steps are always identical
    """
    assert isinstance(random_state, int)
    assert isinstance(df, pd.DataFrame)

    # Fix dataframe
    df = _init_df(df)

    # Fix random number generator
    rng = init_rng(rng=rng, random_state=random_state)
    return rng, df


def _init_df(df):
    columns = df.columns.tolist()

    if columns[-1] == "a_lbl":
        return df
    else:
        df = _init_anomaly_column(df, name="a_lbl", value=0)
        return df


def _init_anomaly_column(df, name="a_lbl", value=np.nan):
    df[name] = value
    return df


# rng
def init_rng(rng=None, random_state=42):
    if rng is None:
        return np.random.default_rng(random_state)
    else:
        return rng


# Location in dataframe
def init_row_idx(rng, df, row=None):
    return _init_idx(rng, df, row_or_col=row, kind="row")


def init_col_idx(rng, df, col=None):
    return _init_idx(rng, df, row_or_col=col, kind="col")


def _init_idx(rng, df, row_or_col=None, kind="row"):
    axes = dict(row=0, col=1)
    axis = axes[kind]

    if row_or_col is None:
        return rng.integers(df.shape[axis], dtype=int)
    elif isinstance(row_or_col, str) and kind in {"col"}:
        assert (
            row_or_col in df.columns
        ), "If you give a string, this column needs to present in the DataFrame."
        return df.columns.get_loc(row_or_col)
    elif isinstance(row_or_col, str) and kind in {"row"}:
        assert (
            row_or_col in df.index
        ), "If you give a string, this row needs to present in the index of the DataFrame."
        return df.index.get_loc(row_or_col)
    else:
        assert isinstance(row_or_col, int)
        assert row_or_col < df.shape[axis], "Size must be feasible."

        return row_or_col


# Draw
def draw_val(rng, val_list, val_dist):
    assert isinstance(val_list, np.ndarray)

    return rng.choice(val_list, p=val_dist)


# Values list
def init_val_list(df, col_idx, val_list=None):

    if val_list is None:
        # Take all the given values as possible options.
        val_list = df.iloc[:, col_idx].values
    else:
        assert isinstance(val_list, np.ndarray)
        # TODO: maybe assert types of elements
        val_list = val_list
    return val_list


# Dist
def init_val_dist(df, col_idx, val_list, val_dist=None):
    assert isinstance(val_list, np.ndarray)
    if val_dist in {None, "uniform"}:
        val_dist = _uniform_dist(val_list)
    else:
        assert isinstance(val_dist, np.ndarray)
        assert val_dist.shape == val_list.shape
    return val_dist


def _uniform_dist(val_list):
    assert isinstance(val_list, np.ndarray)
    n_vals = val_list.shape[0]
    assert n_vals > 0, "val_list needs to contain values."
    return np.ones(n_vals) * (1 / n_vals)
