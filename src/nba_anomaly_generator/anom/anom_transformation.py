import numpy as np
import pandas as pd

from .utils import assert_and_rng_and_prep_df, init_col_idx, init_row_idx


def insert_transformation_anomaly(
    df,
    row=None,
    col=None,
    transformation=None,
    rng=None,
    random_state=42,
    return_anomaly_metadata=True,
    **transformation_kwargs,
):
    """
    Dependency anomaly = Feature value that does not follow the dependency patterns present in the data.
    
    transformation: function that takes in the old value and some kwargs
    
    E.g.
        Replace the weight of a player by the weight of another player. 
        
        The fake value will be 'realistic', globally speaking, but will not 
        follow any dependency pattern between features of the same instance (unless by sheer chance.)
    """

    rng, df = assert_and_rng_and_prep_df(df, rng=rng, random_state=random_state)

    # Fix location
    row_idx = init_row_idx(rng, df, row=row)
    col_idx = init_col_idx(rng, df, col=col)

    # Fix transformation
    transformation = _init_transformation(transformation=transformation)

    # Old, New value + Substitution
    old = df.iat[row_idx, col_idx]
    new = transformation(old, **transformation_kwargs)
    df.iat[row_idx, col_idx] = new

    # N.b. the _init_df method garantuees last column is the anomaly one.
    df.iat[row_idx, -1] = 1

    if return_anomaly_metadata:
        # anomaly metadata
        anomaly_metadata = dict(
            iloc=(row_idx, col_idx),
            loc=(df.index[row_idx], df.columns[col_idx]),
            old=old,
            new=new,
        )
        return df, anomaly_metadata
    else:
        return df


# Some included transformations
def _init_transformation(transformation=None):
    if transformation is None:
        # default transformation is times 2
        return lambda x: 2 * x
    else:
        return transformation


def lb_to_kg(weight_in_lb):
    if not isinstance(weight_in_lb, (int, float)):
        weight_in_lb = float(weight_in_lb)
    POUND_IN_KILOGRAM = 0.45359237
    return POUND_IN_KILOGRAM * weight_in_lb


def ft_to_m(height_in_ft):
    if isinstance(height_in_ft, (int, float)) and height_in_ft < 9:
        feet = height_in_ft
        inch = 0
    elif isinstance(height_in_ft, tuple):
        feet = height_in_ft[0]
        inch = height_in_ft[1]
    elif isinstance(height_in_ft, str):
        feet, inch = _parse_height_in_ft(height_in_ft)
    else:
        raise ValueError("I cannot handle this: {}".format(height_in_ft))

    FOOT_IN_M = 0.3048
    INCH_IN_M = 0.0254
    return FOOT_IN_M * feet + INCH_IN_M * inch


def _parse_height_in_ft(height_in_ft_str):
    feet, inch = height_in_ft_str.split("-")
    return int(feet), int(inch)
