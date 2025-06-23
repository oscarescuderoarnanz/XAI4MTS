# Required Libraries
import numpy as np  # For numerical computations and array manipulations
import pandas as pd  # For handling dataframes
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Masking,
    Dense,
    LayerNormalization,
    Dropout,
    Add,
    TimeDistributed,
    MultiHeadAttention
)
from tensorflow.keras import backend as K
from joblib import Parallel, delayed  # For parallel computation
import multiprocessing

import sys
sys.path.append("../")
from utils import *
from explainability_methods.att_method import hadamard_attention


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Implements sinusoidal positional encoding as described
    in "Attention Is All You Need".
    """
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
        Generates a (position, d_model) matrix using sine and cosine functions.
        """
        angle_rads = self._get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model
        )  # shape = (position, d_model)

        sines = tf.math.sin(angle_rads[:, 0::2])
        coses = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.reshape(
            tf.concat([sines, coses], axis=-1),
            [position, d_model]
        )
        return pos_encoding  # shape = (position, d_model)

    def _get_angles(self, pos, i, d_model):
        """
        Computes the angle rates for positional encoding.

        Args:
            pos: Tensor of shape (position, 1)
            i: Tensor of shape (1, d_model)
            d_model: Integer representing model dimensionality
        """
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, inputs):
        """
        Adds positional encoding to the input tensor.

        Args:
            inputs: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of the same shape with positional encodings added.
        """
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.pos_encoding[tf.newaxis, :seq_len, :]  # shape = (1, seq_len, d_model)
        return inputs + pos_encoding  # Broadcasting over batch_size


def build_model_Transformer(hyperparameters):
    """
    Builds a Transformer Encoder model with Hadamard attention and masking.

    The function applies Masking to ignore any values equal to `mask_value`,
    then computes Hadamard attention on the masked tensor, projects to d_model,
    adds positional encoding, and stacks multiple Transformer encoder blocks.

    Args:
        hyperparameters: Dictionary containing the following keys:
            - n_time_steps: Integer number of time steps (sequence length).
            - layers: List of three integers [input_dim, d_model, num_heads].
                      * input_dim: Dimensionality of each input feature vector.
                      * d_model: Dimensionality inside the Transformer.
                      * num_heads: Number of attention heads.
            - mask_value: Scalar value to be masked (e.g., 666).
            - dropout: Float dropout rate for both Hadamard and Transformer attention.
            - ff_dim: Integer feed-forward inner dimension.
            - num_transformer_blocks: Integer number of Transformer encoder blocks.
            - activation: String activation function for the feed-forward layers (e.g., 'relu').
            - lr_scheduler: Float learning rate for the optimizer.

    Returns:
        model: A compiled tf.keras.Model instance.
        hadamard_scores: A tf.Tensor containing Hadamard attention scores.
        dynamic_input: The Input layer tensor.
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

    # (1) Input layer and masking
    dynamic_input = Input(shape=(seq_len, dim_input), name="input_sequence")
    masked = Masking(mask_value=mask_value, name="masking_layer")(dynamic_input)
    # The Masking layer will ignore any timesteps where the value equals mask_value.

    # (2) Hadamard attention on the masked tensor
    weighted, hadamard_scores = hadamard_attention(
        masked,
        dim_input,
        dropout_rate
    )
    # weighted: Tensor of shape (batch_size, seq_len, dim_input)
    # hadamard_scores: Tensor of shape (batch_size, seq_len, dim_input) or similar

    # (3) Project to d_model if necessary
    if dim_input != d_model:
        x = Dense(
            d_model,
            activation=None,
            use_bias=False,
            name="input_projection"
        )(weighted)
    else:
        x = weighted  # If input dimension already equals d_model, skip projection

    # (4) Add positional encoding
    x = PositionalEncoding(seq_len, d_model)(x)  # shape = (batch_size, seq_len, d_model)

    # (5) Define a single Transformer Encoder block
    def transformer_encoder_block(x_in, block_idx):
        """
        Applies one Transformer encoder block consisting of:
          1) LayerNormalization
          2) MultiHeadAttention
          3) Dropout + Residual
          4) LayerNormalization
          5) Feed-forward (Dense(ff_dim) -> activation -> Dense(d_model))
          6) Dropout + Residual

        Args:
            x_in: Input tensor of shape (batch_size, seq_len, d_model).
            block_idx: Integer block index used for unique naming.

        Returns:
            Tensor of shape (batch_size, seq_len, d_model).
        """
        # 5.1 LayerNorm prior to attention
        x_norm1 = LayerNormalization(epsilon=1e-6, name=f"ln1_block{block_idx}")(x_in)

        # 5.2 MultiHead self-attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f"multihead_attn_block{block_idx}"
        )(x_norm1, x_norm1)
        attn_output = Dropout(dropout_rate, name=f"dropout_attn_block{block_idx}")(attn_output)

        # Residual connection
        out1 = Add(name=f"residual_attn_block{block_idx}")([x_in, attn_output])

        # 5.3 LayerNorm prior to feed-forward
        x_norm2 = LayerNormalization(epsilon=1e-6, name=f"ln2_block{block_idx}")(out1)

        # 5.4 Feed-forward sublayer
        ffn = Dense(ff_dim, activation=activation, name=f"ffn_dense1_block{block_idx}")(x_norm2)
        ffn = Dense(d_model, name=f"ffn_dense2_block{block_idx}")(ffn)
        ffn = Dropout(dropout_rate, name=f"dropout_ffn_block{block_idx}")(ffn)

        # Second residual connection
        out2 = Add(name=f"residual_ffn_block{block_idx}")([out1, ffn])

        return out2

    # (6) Stack multiple Transformer encoder blocks
    for i in range(num_blocks):
        x = transformer_encoder_block(x, block_idx=i)

    # (7) Output layer: TimeDistributed Dense with sigmoid activation
    output = TimeDistributed(
        Dense(1, activation="sigmoid", use_bias=False),
        name="output_layer"
    )(x)
    # Output shape: (batch_size, seq_len, 1)

    model = Model(inputs=dynamic_input, outputs=output, name="Transformer_with_Hadamard")
    model.compile(
        loss='binary_crossentropy',
        sample_weight_mode="temporal",  # Ensures sample_weight is applied per timestep
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters["lr_scheduler"]),
        metrics=['accuracy', 'AUC'],
        weighted_metrics=[]  # Prevents warnings when sample_weight is provided without weighted_metrics
    )

    return model, hadamard_scores, dynamic_input


def run_network(
    X_train,
    X_val,
    y_train,
    y_val,
    sample_weights_train,
    sample_weights_val,
    hyperparameters,
    seed
):
    """
    Trains and evaluates the Transformer model with Hadamard attention.

    Args:
        X_train: NumPy array for training inputs of shape (num_samples, seq_len, input_dim).
        X_val: NumPy array for validation inputs of shape (num_samples, seq_len, input_dim).
        y_train: NumPy array for training labels reshaped to (num_samples, seq_len, 1).
        y_val: NumPy array for validation labels reshaped to (num_samples, seq_len, 1).
        sample_weights_train: NumPy array of sample weights for training of shape (num_samples, seq_len).
        sample_weights_val: NumPy array of sample weights for validation of shape (num_samples, seq_len).
        hyperparameters: Dictionary containing model hyperparameters.
        seed: Integer seed for reproducibility.

    Returns:
        model: A trained tf.keras.Model instance.
        hist: Training history returned by model.fit().
        earlystopping: The EarlyStopping callback instance (with best weights restored).
        hadamard_scores: tf.Tensor of Hadamard attention scores.
        dynamic_input: The Input layer tensor.
    """
    batch_size = hyperparameters['batch_size']
    n_epochs_max = hyperparameters['n_epochs_max']

    model, hadamard_scores, dynamic_input = build_model_Transformer(hyperparameters)

    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=hyperparameters["mindelta"],
        patience=hyperparameters["patience"],
        restore_best_weights=True,
        mode="min"
    )

    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val, sample_weights_val.squeeze()),
        callbacks=[earlystopping],
        batch_size=batch_size,
        epochs=n_epochs_max,
        verbose=hyperparameters["verbose"],
        sample_weight=sample_weights_train.squeeze(),
    )

    return model, hist, earlystopping, hadamard_scores, dynamic_input


def evaluate_combination(
    k,
    l,
    m,
    a,
    b,
    c,
    hyperparameters,
    dropout,
    layers,
    lr_scheduler,
    adjustment_factor,
    activation,
    num_heads,
    seed,
    split,
    norm,
    n_time_steps
):
    """
    Evaluates a single combination of hyperparameters using 5-fold validation.
    Returns the average validation loss across folds.

    Args:
        k: Index for dropout list.
        l: Index for layers list.
        m: Index for learning rate list.
        a: Index for adjustment factor list.
        b: Index for activation list.
        hyperparameters: Base dictionary of hyperparameters.
        dropout: List of candidate dropout rates.
        layers: List of layer configurations ([input_dim, d_model, num_heads]).
        lr_scheduler: List of candidate learning rates.
        adjustment_factor: List of adjustment factor values.
        activation: List of activation function names.
        seed: Integer seed for reproducibility.
        split: String denoting data split subfolder.
        norm: String denoting normalization suffix in filenames.
        n_time_steps: Integer number of time steps.

    Returns:
        A tuple containing:
            (mean_validation_loss, k, l, m, a, b, X_train, y_train, X_val, y_val)
    """
    hyper_copy = hyperparameters.copy()
    hyper_copy['dropout'] = dropout[k]
    hyper_copy['layers'] = layers[l]
    hyper_copy['lr_scheduler'] = lr_scheduler[m]
    hyper_copy['adjustment_factor'] = adjustment_factor[a]
    hyper_copy['activation'] = activation[b]
    hyper_copy['num_heads'] = num_heads[c]

    dataset = hyper_copy['dataset']
    validation_losses = []

    for i in range(5):
        print("cv:", i)
        X_train = np.load(f"../../../../DATA/{dataset}/{split}/X_train_tensor_{i}{norm}.npy")
        y_train = pd.read_csv(f"../../../../DATA/{dataset}/{split}/y_train_tensor_{i}{norm}.csv")
        X_val = np.load(f"../../../../DATA/{dataset}/{split}/X_val_tensor_{i}{norm}.npy")
        y_val = pd.read_csv(f"../../../../DATA/{dataset}/{split}/y_val_tensor_{i}{norm}.csv")

        reset_keras()
        sample_weights_train = create_temp_weight_mod(y_train, hyper_copy, timeSteps=n_time_steps)
        sample_weights_val = create_temp_weight_mod(y_val, hyper_copy, timeSteps=n_time_steps)

        model, hist, early, hadamard_scores, dynamic_input = run_network(
            X_train,
            X_val,
            y_train.loc[:, 'individualMRGerm'].values.reshape(y_train.shape[0] // n_time_steps, n_time_steps, 1),
            y_val.loc[:, 'individualMRGerm'].values.reshape(y_val.shape[0] // n_time_steps, n_time_steps, 1),
            sample_weights_train,
            sample_weights_val,
            hyper_copy,
            seed
        )

        validation_losses.append(np.min(hist.history["val_loss"]))

    mean_val_loss = np.mean(validation_losses)
    return (mean_val_loss, k, l, m, a, b, c, X_train, y_train, X_val, y_val)


def myCVGridParallel(
    hyperparameters,
    dropout,
    lr_scheduler,
    layers,
    adjustment_factor,
    activation,
    num_heads,
    seed,
    split,
    norm,
    n_time_steps
):
    """
    Performs a parallelized grid search over all hyperparameter combinations.
    Returns the best hyperparameters (lowest average validation loss).

    Args:
        hyperparameters: Base dictionary of hyperparameters.
        dropout: List of candidate dropout rates.
        lr_scheduler: List of candidate learning rates.
        layers: List of layer configurations ([input_dim, d_model, num_heads]).
        adjustment_factor: List of adjustment factor values.
        activation: List of activation function names.
        seed: Integer seed for reproducibility.
        split: String denoting data split subfolder.
        norm: String denoting normalization suffix in filenames.
        n_time_steps: Integer number of time steps.

    Returns:
        Dictionary containing the best hyperparameters and corresponding data:
            {
              'dropout': ...,
              'layers': ...,
              'lr_scheduler': ...,
              'adjustment_factor': ...,
              'activation': ...,
              'X_train': ...,
              'y_train': ...,
              'X_val': ...,
              'y_val': ...
            }
    """
    best_hyperparameters = {}
    best_metric = np.inf

    results = Parallel(n_jobs=32)(
        delayed(evaluate_combination)(
            k, l, m, a, b, c,
            hyperparameters,
            dropout,
            layers,
            lr_scheduler,
            adjustment_factor,
            activation,
            num_heads,
            seed,
            split,
            norm,
            n_time_steps
        )
        for k in range(len(dropout))
        for l in range(len(layers))
        for m in range(len(lr_scheduler))
        for a in range(len(adjustment_factor))
        for b in range(len(activation))
        for c in range(len(num_heads))
    )

    for mean_val_loss, k, l, m, a, b, c, X_train, y_train, X_val, y_val in results:
        if mean_val_loss < best_metric:
            print("Updating best_metric from", best_metric, "to", mean_val_loss)
            best_metric = mean_val_loss
            best_hyperparameters = {
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

    return best_hyperparameters
