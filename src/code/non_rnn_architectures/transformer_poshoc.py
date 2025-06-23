# Required Libraries
import numpy as np  # For numerical computations and array manipulations
import pandas as pd  # For handling dataframes
import tensorflow as tf
from tensorflow.keras.layers import  Dense, SimpleRNN
from tensorflow.keras import backend as K
from joblib import Parallel, delayed  # For parallel computation
import multiprocessing

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Masking, Dense, LayerNormalization,
    Dropout, Add, TimeDistributed, MultiHeadAttention
)
from tensorflow.keras import Model

import sys
sys.path.append("../")
from utils import *

# (1) Positional Encoding layer (using standard sine/cosine)
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.position = position
        self.d_model   = d_model
        self.pos_encoding = self._build_positional_encoding(position, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "position": self.position,
            "d_model":   self.d_model,
        })
        return config

    def _build_positional_encoding(self, position, d_model):
        """
        Generates a (position, d_model) matrix with
        sine/cosine positional encoding.
        """
        angle_rads = self._get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model
        )  # shape = (position, d_model)

        # Apply sin to even indices (2i) and cos to odd indices (2i+1)
        sines = tf.math.sin(angle_rads[:, 0::2])
        coses = tf.math.cos(angle_rads[:, 1::2])

        # Interleave sines and coses back into (position, d_model)
        pos_encoding = tf.reshape(
            tf.concat([sines, coses], axis=-1),
            [position, d_model]
        )
        return pos_encoding  # shape = (position, d_model)

    def _get_angles(self, pos, i, d_model):
        """
        pos: shape = (position, 1)
        i:   shape = (1, d_model)
        """
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, inputs):
        """
        inputs: tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = tf.shape(inputs)[1]
        # Take the positional encodings for the first seq_len positions
        pos_encoding = self.pos_encoding[tf.newaxis, :seq_len, :]  # shape = (1, seq_len, d_model)
        return inputs + pos_encoding  # broadcast over batch_size
    

def build_model_Transformer(hyperparameters):
    """
    Builds a Transformer-Encoder–based model for sequential prediction.
    Assumes hyperparameters contains:
      - n_time_steps: number of time steps (seq_len)
      - layers: [input_dim, d_model]
      - mask_value: value used for masking padded positions
      - dropout: global dropout rate
      - num_heads: number of attention heads
      - ff_dim: feed-forward inner dimension
      - num_transformer_blocks: how many encoder blocks to stack
      - activation: activation function in the inner feed-forward (e.g. 'relu')
      - lr_scheduler: learning rate
    """

    seq_len      = hyperparameters["n_time_steps"]
    dim_input    = hyperparameters["layers"][0]  # dimension of each input vector
    d_model      = hyperparameters["layers"][1]  # model dimension for Transformer
    num_heads    = hyperparameters["num_heads"]
    ff_dim       = hyperparameters["ff_dim"]
    num_blocks   = hyperparameters["num_transformer_blocks"]
    dropout_rate = hyperparameters["dropout"]
    activation   = hyperparameters["activation"]
    mask_value   = hyperparameters["mask_value"]

    # (A) Dynamic input: shape = (batch_size, seq_len, dim_input)
    dynamic_input = Input(shape=(seq_len, dim_input))

    # (B) Masking layer for positions equal to mask_value (to handle padding)
    x = Masking(mask_value=mask_value)(dynamic_input)  # shape = (batch, seq_len, dim_input)

    # (C) If dim_input != d_model, project input to d_model via a Dense layer
    if dim_input != d_model:
        x = Dense(d_model, activation=None, use_bias=False, name="input_projection")(x)
        # Now x has shape = (batch, seq_len, d_model)
    # If dim_input == d_model, skip this projection

    # (D) Add Positional Encoding
    x = PositionalEncoding(seq_len, d_model)(x)  # shape = (batch, seq_len, d_model)

    # (E) Define a single Transformer Encoder block
    def transformer_encoder_block(x_in):
        # 1. Pre-attention LayerNormalization
        x_norm1 = LayerNormalization(epsilon=1e-6)(x_in)
        # 2. Multi-Head Self-Attention (queries=keys=values = x_norm1)
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            # name="multihead_attn"
        )(x_norm1, x_norm1)
        # 3. Dropout + Residual connection
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = Add()([x_in, attn_output])  # Residual: add input to attention output

        # 4. Pre-FFN LayerNormalization
        x_norm2 = LayerNormalization(epsilon=1e-6)(out1)
        # 5. Feed-forward sublayer: Dense(ff_dim) → activation → Dense(d_model)
        ffn = Dense(ff_dim, activation=activation)(x_norm2)
        ffn = Dense(d_model)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        # 6. Residual connection again
        out2 = Add()([out1, ffn])

        return out2

    # (F) Stack `num_blocks` Transformer Encoder blocks
    for i in range(num_blocks):
        x = transformer_encoder_block(x)

    # (G) Output layer: TimeDistributed(Dense(1, activation="sigmoid"))
    output = TimeDistributed(Dense(1, activation="sigmoid", use_bias=False))(x)
    # Output shape = (batch, seq_len, 1)

    model = Model(inputs=dynamic_input, outputs=output)

    # (H) Compile the model
    model.compile(
        loss='binary_crossentropy',
        sample_weight_mode="temporal",
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters["lr_scheduler"]),
        metrics=['accuracy', 'AUC'],
        weighted_metrics=[] 
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

    model = build_model_Transformer(hyperparameters)
        
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




def evaluate_combination(k, l, m, a, b, c, hyperparameters, dropout, layers, lr_scheduler, adjustment_factor, activation, num_heads, seed, split, norm, n_time_steps):
    hyperparameters_copy = hyperparameters.copy()
    hyperparameters_copy['dropout'] = dropout[k]
    hyperparameters_copy['layers'] = layers[l]
    hyperparameters_copy['lr_scheduler'] = lr_scheduler[m]
    hyperparameters_copy['adjustment_factor'] = adjustment_factor[a]
    hyperparameters_copy['activation'] = activation[b]
    hyperparameters_copy['num_heads'] = num_heads[c]
    
    dataset = hyperparameters_copy['dataset']
    
    v_val_loss = []

    for i in range(5):
        print("i:", i)
        X_train = np.load(f"../../../../DATA/{dataset}/{split}/X_train_tensor_{i}{norm}.npy")
        y_train = pd.read_csv(f"../../../../DATA/{dataset}/{split}/y_train_tensor_{i}{norm}.csv")
        X_val = np.load(f"../../../../DATA/{dataset}/{split}/X_val_tensor_{i}{norm}.npy")
        y_val = pd.read_csv(f"../../../../DATA/{dataset}/{split}/y_val_tensor_{i}{norm}.csv")

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
    return (metric_dev, k, l, m, a, b, c, X_train, y_train, X_val, y_val)

def myCVGridParallel(hyperparameters, dropout, lr_scheduler, layers, adjustment_factor, activation, num_heads, seed, split, norm, n_time_steps):
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
    results = Parallel(n_jobs=32)(
        delayed(evaluate_combination)(k, l, m, a, b, c, hyperparameters, dropout, layers, lr_scheduler, adjustment_factor, activation, num_heads, seed, split, norm, n_time_steps)
        for k in range(len(dropout))
        for l in range(len(layers))
        for m in range(len(lr_scheduler))
        for a in range(len(adjustment_factor))
        for b in range(len(activation))
        for c in range(len(num_heads))
    )

    for metric_dev, k, l, m, a, b, c, X_train, y_train, X_val, y_val in results:
        if metric_dev < bestMetricDev:
            print("\t\t\tCambio the best", bestMetricDev, "por metric dev:", metric_dev)
            bestMetricDev = metric_dev
            bestHyperparameters = {
                'dropout': dropout[k],
                'layers': layers[l],
                'lr_scheduler': lr_scheduler[m],
                'adjustment_factor': adjustment_factor[a],
                'activation': activation[b],
                'num_heads': num_heads[c],
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }

    return bestHyperparameters