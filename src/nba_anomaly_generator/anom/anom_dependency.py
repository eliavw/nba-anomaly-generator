import numpy as np
import pandas as pd

from .utils import (
    assert_and_rng_and_prep_df,
    draw_val,
    init_col_idx,
    init_row_idx,
    init_val_dist,
    init_val_list,
)


def insert_dependency_anomaly(
    df,
    row=None,
    col=None,
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
    col_idx = init_col_idx(rng, df, col=col)

    # Fix value
    val = _get_val_dependency_anomaly(
        rng, df, col_idx, val=val, val_list=val_list, val_dist=val_dist
    )

    # Insert value and label
    old = df.iat[row_idx, col_idx]
    df.iat[row_idx, col_idx] = val

    # N.b. the _init_df method garantuees last column is the anomaly one.
    df.iat[row_idx, -1] = 1

    if return_anomaly_metadata:
        # anomaly metadata
        anomaly_metadata = dict(
            iloc=(row_idx, col_idx),
            loc=(df.index[row_idx], df.columns[col_idx]),
            old=old,
            new=val,
        )
        return df, anomaly_metadata
    else:
        return df


def _get_val_dependency_anomaly(rng, df, col, val=None, val_list=None, val_dist=None):
    assert isinstance(col, int)
    assert isinstance(df, pd.DataFrame)

    if val is None:
        val_list = init_val_list(df, col, val_list=val_list)
        val_dist = init_val_dist(df, col, val_list, val_dist=val_dist)

        return draw_val(rng, val_list, val_dist)
    else:
        return val
