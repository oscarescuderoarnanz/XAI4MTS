import pandas as pd
import numpy as np
from typing import List, Union

def calc_avg_event(data: Union[pd.DataFrame, np.ndarray],
                   numerical_feats: List[Union[str, int]],
                   model_features: List[str] = None,
                   ) -> pd.DataFrame:
    """
    Calculates the average event of a dataset. This event is repeated N times
    to form the background sequence to be used in TimeSHAP.

    Calculates the median of numerical features of a pandas DataFrame

    Parameters
    ----------
    data: pd.DataFrame
        Dataset to use for baseline calculation

    numerical_feats: List[Union[str, int]]
        List of numerical features or corresponding indexes to calculate median of

    model_features: List[str]
        Model features to infer the indexes of schema. Needed when using strings to identify features

    Returns
    -------
    pd.DataFrame
        DataFrame with the median of the features
    """
    if len(numerical_feats) > 0 and isinstance(numerical_feats[0], str):
        # given features are not indexes
        if isinstance(data, pd.DataFrame):
            model_features = list(data.columns)
        else:
            assert model_features is not None and len(model_features), "When using feature names to identify them, specify the model features. Alternatively you can pass the indexes of the features directly"
        numerical_indexes = [model_features.index(x) for x in numerical_feats]
        ordered_feats = numerical_feats
    else:
        numerical_indexes = numerical_feats
        ordered_feats = numerical_indexes

    if len(data.shape) == 3:
        data = np.squeeze(data, axis=1)
    elif len(data.shape) == 2:
        data = data.values
    else:
        raise ValueError("Invalid data shape")

    # Convert to float for nan operations
    num_data = data[:, numerical_indexes].astype(float)
    num_data[num_data == 666] = np.nan
    numerical = np.nanmean(num_data, axis=0)

    return pd.DataFrame([numerical], columns=ordered_feats)



def calc_avg_sequence(data: Union[pd.DataFrame, np.ndarray],
                      numerical_feats: List[Union[str, int]],
                      model_features: List[str] = None,
                      entity_col: str = None,
                      ) -> np.ndarray:
    """
    Calculates the average sequence of a dataset. Requires all sequences of the
    dataset to be the same size and ordered by time.

    Calculates the median of numerical features of a pandas DataFrame.

    Parameters
    ----------
    data: Union[pd.DataFrame, np.ndarray]
        Dataset to use for baseline calculation

    numerical_feats: List[Union[str, int]]
        List of numerical features or corresponding indexes to calculate median of

    model_features: List[str]
        Model features to infer the indexes of schema. Needed when using strings to identify features

    entity_col: str
        Entity column to identify sequences

    Returns
    -------
    np.ndarray
        Average sequence to use in TimeSHAP
    """
    if isinstance(data, pd.DataFrame):
        assert entity_col is not None, "To calculate average sequence from DataFrame, entity_col is required"
        sequences = data.groupby(entity_col)
        seq_lens = sequences.size().values
        assert np.array([x == seq_lens[0] for x in seq_lens]).all(), "All sequences must be the same length"
        data = np.array([x[1].values for x in sequences])
    elif isinstance(data, np.ndarray) and len(data.shape) == 3:
        pass
    else:
        raise ValueError("Unrecognized data format")
    
    if len(numerical_feats) > 0 and isinstance(numerical_feats[0], str):
        # given features are not indexes
        assert model_features is not None and len(model_features), "When using feature names to identify them, specify the model features. Alternatively you can pass the indexes of the features directly"
        numerical_indexes = [model_features.index(x) for x in numerical_feats]
    else:
        numerical_indexes = numerical_feats

    # Convert to float for nan operations
    num_data = data[:, :, numerical_indexes].astype(float)
    num_data[num_data == 666] = np.nan
    numerical = np.nanmean(num_data, axis=0)

    return numerical.astype(float)
