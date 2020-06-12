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
    verbose=False,
    swap=False,
):
    """
    Source subpopulation:
        subpopulation from which we sample the anomaly
    Target subpopulation:
        subpopulation(s) that contains the instance which we will alter
                        
    """

    rng, df = assert_and_rng_and_prep_df(df, rng=rng, random_state=random_state)

    # Obtain source subpopulation
    if src_subpop_filter is not None:
        sdf = df[df.apply(src_subpop_filter, axis=1)]
    else:
        sdf = df[~df.apply(tgt_subpop_filter, axis=1)]

    # Obtain target subpopulation
    if tgt_subpop_filter is not None:
        tdf = df[df.apply(tgt_subpop_filter, axis=1)]
    else:
        tdf = df[~df.apply(src_subpop_filter, axis=1)]

    col_idx = init_col_idx(rng, df, col=col)

    # row_idx is sampled from target subpop
    tgt_row_idx = int(draw_val(rng, tdf.index.values, None))

    if swap:
        src_row_idx = int(draw_val(rng, sdf.index.values, None))

        src_val = df.iloc[src_row_idx, col_idx].copy()
        tgt_val = df.iloc[tgt_row_idx, col_idx].copy()

        df.iat[src_row_idx, col_idx] = tgt_val
        df.iat[tgt_row_idx, col_idx] = src_val

        # N.b. the prep_df method garantuees last column is the anomaly one.
        df.iat[src_row_idx, -1] = 1
        df.iat[tgt_row_idx, -1] = 1

        if return_anomaly_metadata:
            # anomaly metadata
            anomaly_metadata_01 = dict(
                iloc=(src_row_idx, col_idx),
                loc=(df.index[src_row_idx], df.columns[col_idx]),
                old=src_val,
                new=tgt_val,
            )

            anomaly_metadata_02 = dict(
                iloc=(tgt_row_idx, col_idx),
                loc=(df.index[tgt_row_idx], df.columns[col_idx]),
                old=tgt_val,
                new=src_val,
            )

            anomaly_metadata = [anomaly_metadata_01, anomaly_metadata_02]
            return df, anomaly_metadata
        else:
            return df
    else:
        # anomalies are sampled from positive subpop
        src_subpop_col_val_list = sdf.iloc[:, col_idx].values

        if verbose:
            print(src_subpop_col_val_list)

        return insert_dependency_anomaly(
            df,
            row=tgt_row_idx,
            col=col_idx,
            val_list=src_subpop_col_val_list,
            val_dist=None,
            rng=rng,
            random_state=random_state,
            return_anomaly_metadata=return_anomaly_metadata,
        )
