import numpy as np
import pandas as pd

from .utils import assert_and_rng_and_prep_df, init_col_idx, init_row_idx


def insert_swap_anomaly(
    df,
    row=None,
    l_col=None,
    r_col=None,
    val=None,
    val_list=None,
    val_dist=None,
    rng=None,
    random_state=42,
    return_anomaly_metadata=True,
):
    """
    Dependency anomaly = Feature value that does not follow the dependency patterns present in the data.
    
    E.g.
        Replace the weight of a player by the weight of another player. 
        
        The fake value will be 'realistic', globally speaking, but will not 
        follow any dependency pattern between features of the same instance (unless by sheer chance.)
    """

    rng, df = assert_and_rng_and_prep_df(df, rng=rng, random_state=random_state)

    # Fix location
    row_idx = init_row_idx(rng, df, row=row)

    l_col_idx = init_col_idx(rng, df, col=r_col)
    r_col_idx = init_col_idx(rng, df, col=l_col)

    # Fix value
    vals = _get_vals_swap_anomaly(df, row_idx, l_col_idx, r_col_idx)

    # Insert value and label
    df.iat[row_idx, r_col_idx] = vals["l"]
    df.iat[row_idx, r_col_idx] = vals["r"]

    # N.b. the _init_df method garantuees last column is the anomaly one.
    df.iat[row_idx, -1] = 1

    if return_anomaly_metadata:
        # anomaly metadata
        anomaly_metadata = [
            dict(loc=(row_idx, l_col_idx), old=vals["r"], new=vals["l"]),
            dict(loc=(row_idx, r_col_idx), old=vals["l"], new=vals["r"]),
        ]
        return df, anomaly_metadata
    else:
        return df


def _get_vals_swap_anomaly(df, row_idx, l_col_idx, r_col_idx):
    assert isinstance(r_col_idx, int)
    assert isinstance(l_col_idx, int)

    l_val = df.iat[row_idx, l_col_idx]
    r_val = df.iat[row_idx, r_col_idx]

    return dict(l=l_val, r=r_val)
