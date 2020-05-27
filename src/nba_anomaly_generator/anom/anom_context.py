import numpy as np
import pandas as pd

from .anom_dependency import insert_dependency_anomaly
from .utils import assert_and_rng_and_prep_df, draw_val, init_col_idx


def insert_contextual_anomaly(
    df,
    row=None,
    col=None,
    src_subpop_filter=None,
    tgt_subpop_filter=None,
    rng=None,
    random_state=200,
    return_anomaly_metadata=True,
):
    """
    Source subpopulation:
        subpopulation from which we sample the anomaly
    Target subpopulation:
        subpopulation(s) that contains the instance which we will alter
                        
    """

    rng, df = assert_and_rng_and_prep_df(df, rng=rng, random_state=random_state)

    if src_subpop_filter is not None:
        sdf = df[df.apply(src_subpop_filter, axis=1)]
    else:
        sdf = df[~df.apply(tgt_subpop_filter, axis=1)]

    if tgt_subpop_filter is not None:
        tdf = df[df.apply(tgt_subpop_filter, axis=1)]
    else:
        tdf = df[~df.apply(src_subpop_filter, axis=1)]

    col_idx = init_col_idx(rng, df, col=col)
    # row_idx is sampled from negative subpop
    row_idx = int(draw_val(rng, tdf.index.values, None))

    # anomalies are sampled from positive subpop
    src_subpop_col_val_list = sdf.iloc[:, col_idx].values
    print(src_subpop_col_val_list)

    return insert_dependency_anomaly(
        df,
        row=row_idx,
        col=col_idx,
        val_list=src_subpop_col_val_list,
        val_dist=None,
        rng=rng,
        random_state=random_state,
        return_anomaly_metadata=return_anomaly_metadata,
    )
