# Required Libraries
import numpy as np  # For numerical computations and array manipulations
import pandas as pd  # For handling dataframes
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, GRU, Dropout, Dense, SimpleRNN, TimeDistributed
from tensorflow.keras import backend as K
from joblib import Parallel, delayed  # For parallel computation
import multiprocessing

from rnns_architectures.utils import *


def build_model_LSTM(hyperparameters):
    """
    Builds a LSTM model based on several hyperparameters.

    Args:
        - hyperparameters: Dictionary containing the hyperparameters. 
    Returns:
        - model: A tf.keras.Model with the compiled model.
    """
    
    dynamic_input = tf.keras.layers.Input(shape=(hyperparameters["n_time_steps"], hyperparameters["layers"][0]))
    masked = tf.keras.layers.Masking(mask_value=hyperparameters['mask_value'])(dynamic_input)

    gru_encoder = tf.keras.layers.LSTM(
        hyperparameters["layers"][1],
        dropout=hyperparameters['dropout'],
        return_sequences=True,
        activation=hyperparameters['activation'],
        use_bias=False
    )(masked)

    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, use_bias=False, activation="sigmoid"))(gru_encoder)

    model = tf.keras.Model(dynamic_input, [output])
    model.compile(
        loss='binary_crossentropy', sample_weight_mode="temporal",
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters["lr_scheduler"]),
        metrics=['accuracy', 'AUC'],
        weighted_metrics = []
        
    )
        
    return model
 
def build_model_Vanilla(hyperparameters):
    """
    Builds a Vanilla RNN model based on several hyperparameters.

    Args:
        - hyperparameters: Dictionary containing the hyperparameters. 
    Returns:
        - model: A tf.keras.Model with the compiled model.
    """
    
    dynamic_input = tf.keras.layers.Input(shape=(hyperparameters["n_time_steps"], hyperparameters["layers"][0]))
    masked = tf.keras.layers.Masking(mask_value=hyperparameters['mask_value'])(dynamic_input)
 
    rnn_encoder = SimpleRNN(
        hyperparameters["layers"][1],
        dropout=hyperparameters['dropout'],
        return_sequences=True,
        activation=hyperparameters['activation'],
        use_bias=False
    )(masked)
 
    output = tf.keras.layers.TimeDistributed(Dense(1, use_bias=False, activation="sigmoid"))(rnn_encoder)
 
    model = tf.keras.Model(inputs=dynamic_input, outputs=output)
    model.compile(
        loss='binary_crossentropy', sample_weight_mode="temporal",
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters["lr_scheduler"]),
        metrics=['accuracy', "AUC"],
        weighted_metrics = []
    )
 
    return model

def build_model_GRU(hyperparameters):
    """
    Builds a GRU model based on several hyperparameters.

    Args:
        - hyperparameters: Dictionary containing the hyperparameters. 
    Returns:
        - model: A tf.keras.Model with the compiled model.
    """
    
    dynamic_input = tf.keras.layers.Input(shape=(hyperparameters["n_time_steps"], hyperparameters["layers"][0]))
    masked = tf.keras.layers.Masking(mask_value=hyperparameters['mask_value'])(dynamic_input)

    gru_encoder = tf.keras.layers.GRU(
        hyperparameters["layers"][1],
        dropout=hyperparameters['dropout'],
        return_sequences=True,
        activation=hyperparameters['activation'],
        use_bias=False
    )(masked)

    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, use_bias=False, activation="sigmoid"))(gru_encoder)

    model = tf.keras.Model(dynamic_input, [output])
    model.compile(
        loss='binary_crossentropy', sample_weight_mode="temporal",
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters["lr_scheduler"]),
        metrics=['accuracy', "AUC"],
        weighted_metrics = []
    )
        
    return model


def run_network(X_train, X_val, y_train, y_val, 
                sample_weights_train, sample_weights_val,
                hyperparameters, seed):
    """
    Trains and evaluates the built GRU model based on the provided data and hyperparameters.

    Args:
        - X_train, X_val, y_train, y_val: numpy.ndarray. Training (T) and Validation (V) data labels.
        - sample_weights_train, sample_weights_val: numpy.ndarray. Weights for the T and V data to handle class imbalance.
        - hyperparameters: Dictionary containing the hyperparameters.
        - seed: Integer seed for reproducibility.
    Returns:
        - model: A tf.keras.Model with the trained model.
        - hist:  The training history.
        - earlystopping: The early stopping callback.
    """
    batch_size = hyperparameters['batch_size']
    n_epochs_max = hyperparameters['n_epochs_max']

    if hyperparameters["model_type"] == 'GRU':
        model = None
        model = build_model_GRU(hyperparameters)
    
    elif hyperparameters["model_type"] == 'LSTM':
        model = None
        model = build_model_LSTM(hyperparameters)
        
    elif hyperparameters["model_type"] == 'Vanilla':
        model = None
        model = build_model_Vanilla(hyperparameters)
    
    else:
        print('Incorrect model name. Try GRU, LSTM or Vanilla')
        
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=hyperparameters["mindelta"],
                                                  patience=hyperparameters["patience"],
                                                  restore_best_weights=True,
                                                  mode="min")
    hist = model.fit(X_train, y_train,
                     validation_data=(X_val, y_val, sample_weights_val.squeeze()),
                     callbacks=[earlystopping], batch_size=batch_size, epochs=n_epochs_max,
                     verbose=hyperparameters['verbose'], sample_weight=sample_weights_train.squeeze())
    
    return model, hist, earlystopping




def evaluate_combination(k, l, m, a, b, hyperparameters, dropout, layers, lr_scheduler, adjustment_factor, activation, seed, split, norm, n_time_steps):
    hyperparameters_copy = hyperparameters.copy()
    hyperparameters_copy['dropout'] = dropout[k]
    hyperparameters_copy['layers'] = layers[l]
    hyperparameters_copy['lr_scheduler'] = lr_scheduler[m]
    hyperparameters_copy['adjustment_factor'] = adjustment_factor[a]
    hyperparameters_copy['activation'] = activation[b]
    
    dataset = hyperparameters_copy['dataset']
    
    v_val_loss = []

    for i in range(5):
        X_train = np.load(f"../../../DATA/{dataset}/{split}/X_train_tensor_{i}{norm}.npy")
        y_train = pd.read_csv(f"../../../DATA/{dataset}/{split}/y_train_tensor_{i}{norm}.csv")
        X_val = np.load(f"../../../DATA/{dataset}/{split}/X_val_tensor_{i}{norm}.npy")
        y_val = pd.read_csv(f"../../../DATA/{dataset}/{split}/y_val_tensor_{i}{norm}.csv")

        reset_keras()
        sample_weights_train = create_temp_weight_mod(y_train, hyperparameters_copy, timeSteps=n_time_steps)
        sample_weights_val = create_temp_weight_mod(y_val, hyperparameters_copy, timeSteps=n_time_steps)

        model, hist, early = run_network(
            X_train, X_val,
            y_train.loc[:, 'individualMRGerm'].values.reshape(y_train.shape[0] // n_time_steps, n_time_steps, 1),
            y_val.loc[:, 'individualMRGerm'].values.reshape(y_val.shape[0] // n_time_steps, n_time_steps, 1),
            sample_weights_train, sample_weights_val,
            hyperparameters_copy,
            seed
        )

        v_val_loss.append(np.min(hist.history["val_loss"]))

    metric_dev = np.mean(v_val_loss)
    return (metric_dev, k, l, m, a, b, X_train, y_train, X_val, y_val)

def myCVGridParallel(hyperparameters, dropout, lr_scheduler, layers, adjustment_factor, activation, seed, split, norm, n_time_steps):
    """Parallelized Grid Search. 
       Calculate metricDev based on the evaluation. Compares the metricDev with the current bestMetricDev. 
       If better, updates bestMetricDev and stores those hyperparameters in bestHyperparameters.
       
    Args:
        - hyperparameters: Dictionary containing the hyperparameters.
        - dropout: A list of dropout rates.
        - lr_scheduler: A list of learning rates.
        - layers: A list of layer configurations.
        - seed : Seed value for reproducibility.
        - split: String indicating the data split.
        - norm: String with the type of normalization applied to the data.
    Returns:
        - bestHyperparameters: A dictionary with the best hyperparameters found and Train and Val data.
    """
    bestHyperparameters = {}
    bestMetricDev = np.inf

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=12)(
        delayed(evaluate_combination)(k, l, m, a, b, hyperparameters, dropout, layers, lr_scheduler, adjustment_factor, activation, seed, split, norm, n_time_steps)
        for k in range(len(dropout))
        for l in range(len(layers))
        for m in range(len(lr_scheduler))
        for a in range(len(adjustment_factor))
        for b in range(len(activation))
    )

    for metric_dev, k, l, m, a, b, X_train, y_train, X_val, y_val in results:
        if metric_dev < bestMetricDev:
            print("\t\t\tCambio the best", bestMetricDev, "por metric dev:", metric_dev)
            bestMetricDev = metric_dev
            bestHyperparameters = {
                'dropout': dropout[k],
                'layers': layers[l],
                'lr_scheduler': lr_scheduler[m],
                'adjustment_factor': adjustment_factor[a],
                'activation': activation[b],
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }

    return bestHyperparameters