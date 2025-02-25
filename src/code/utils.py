# Required Libraries
import numpy as np  # For numerical computations and array manipulations
import pandas as pd  # For handling dataframes
import tensorflow as tf
from tensorflow.keras import backend as K
import random, os, json
import pickle


def create_temp_weight_mod(y, hyperparameters, timeSteps):
    """
    Create temporary weights based on class imbalance in the data, based on the frequency of AMR and non-AMR over time.

    Args:
    - y: A DataFrame containing the target variable and additional columns 'Admissiondboid' and 'dayToDone'.
    - hyperparameters: A dictionary containing the model's hyperparameters.
    - timeSteps: An integer representing the number of time steps.
    Returns:
    - A 3D numpy array with the calculated weights as (samples, timeSteps, 1).
    """
   
    df_imbalance = y
    df_imbalance = df_imbalance[df_imbalance.individualMRGerm != hyperparameters["mask_value"]][["dayToDone", "individualMRGerm"]]
    arr_imbalance = df_imbalance.groupby("dayToDone").mean().values
    
    # Invert frequencies to obtain penalty
    arr_penalty_positives = 1 / arr_imbalance
    arr_penalty_negatives = 1 / (1 - arr_imbalance)

    # Apply adjustment factor
    arr_penalty_positives *= hyperparameters['adjustment_factor']
    
    y_func = y.loc[:, 'individualMRGerm'].values.reshape(y.shape[0] // timeSteps, timeSteps, 1)
    
    weights = np.ones(y_func.shape)
    for t in range(timeSteps):
        weights[:, t, :] = np.where(y_func[:, t, :] == 1, arr_penalty_positives[t], arr_penalty_negatives[t])
    weights = np.where(y_func == hyperparameters["mask_value"], 0, weights)
        
    return weights



def reset_keras(seed=42):
    """Function to ensure that results from Keras models
    are consistent and reproducible across different runs"""
    
    K.clear_session()
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)
    
    
def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    