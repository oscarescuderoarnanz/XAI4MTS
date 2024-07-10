from tensorflow.keras import backend as K
import numpy as np
import random, os, json
import tensorflow as tf

### RESET KERAS ###
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
    
    
    
### BBCE TEMPORAL ###
def create_temp_weight(y, hyperparameters, timeSteps=14):
    """
    Creates temporary weights based on class imbalance in the data, based on the frequency of AMR and non-AMR over time.

    Args:
        - y: A DataFrame containing the target variable and additional columns 'Admissiondboid' and 'dayToDone'.
        - hyperparameters: A dictionary containing model hyperparameters.
        - timeSteps: An integer. Is the number of time steps.
    Returns:
        - A 3D numpy array with the computed weights as (samples, timeSteps, 1).
    """
   
    df_imbalance = y
    df_imbalance = df_imbalance[df_imbalance.individualMRGerm != hyperparameters["mask_value"]][["dayToDone", "individualMRGerm"]]
    arr_imbalance = df_imbalance.groupby("dayToDone").mean().values
    arr_penalty_positives = 1 / arr_imbalance
    arr_penalty_negatives = 1 / (1-arr_imbalance)

    y_func = y.loc[:, 'individualMRGerm'].values.reshape(y.shape[0] // timeSteps, timeSteps, 1)
    
    weights = np.ones(y_func.shape)
    weights = np.where(y_func == 1, arr_penalty_positives, arr_penalty_negatives)
    weights = np.where(y_func == hyperparameters["mask_value"], 0, weights)
        
    return weights