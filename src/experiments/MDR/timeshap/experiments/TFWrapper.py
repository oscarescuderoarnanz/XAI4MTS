import numpy as np
import tensorflow as tf
from typing import Tuple
import math
import copy
import pandas as pd

class KerasModelWrapper:
    """Wrapper for Keras machine learning models.

    Encompasses necessary logic to utilize Keras models as lambda functions
    required for TimeSHAP explanations.

    This wrapper is responsible to create tensors, batching processes, and obtaining predictions from tensors.

    Attributes
    ----------
    model: tf.keras.Model
        Keras model to wrap. This model is required to receive a tf.Tensor
        of sequences and returning the score for each instance of each sequence.

    batch_budget: int
        The number of instances to score at a time. Needed to not overload
        GPU memory. Default is 750K. Equates to a 7GB batch

    device: str
        Device to use for computation (e.g., 'cpu:0', 'gpu:0')

    Methods
    -------
    predict_last(data: pd.DataFrame, metadata: Matrix) -> list
        Creates explanations for each instance in ``data``.
    """
    def __init__(self, model: tf.keras.Model, batch_budget: int = 750000, device: str = '/cpu:0'):
        self.model = model
        self.batch_budget = batch_budget
        self.device = device

    def prepare_input(self, input):
        sequence = copy.deepcopy(input)
        if isinstance(sequence, pd.DataFrame):
            sequence = np.expand_dims(sequence.values, axis=0)
        elif len(sequence.shape) == 2 and isinstance(sequence, np.ndarray):
            sequence = np.expand_dims(sequence, axis=0)

        if not (len(sequence.shape) == 3 and isinstance(sequence, np.ndarray)):
            raise ValueError("Input type not supported")

        # Ignore elements with value 666 by setting them to 0 (or another placeholder value)
        sequence[sequence == 666] = 0

        return sequence

    def predict_last_hs(self, sequences: np.ndarray, hidden_states: np.ndarray = None) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
        sequences = self.prepare_input(sequences)

        sequence_len = sequences.shape[1]
        batch_size = math.floor(self.batch_budget / sequence_len)
        batch_size = max(1, batch_size)

        # Count the number of rows with values other than 666 in the input x
        num_valid_rows = np.sum(np.any(sequences != 666, axis=1))

        return_scores = []
        return_hs = []
        hs = None

        for i in range(0, num_valid_rows, batch_size):
            batch = sequences[i:(i + batch_size), :, :]
            batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            with tf.device(self.device):
                if hidden_states is not None:
                    hidden_states_tensor = [
                        tf.convert_to_tensor(hs_layer[i:(i + batch_size), :], dtype=tf.float32) 
                        if len(hs_layer.shape) == 2 else 
                        tf.convert_to_tensor(hs_layer[:, i:(i + batch_size), :], dtype=tf.float32) 
                        for hs_layer in hidden_states
                    ]
                    predictions = self.model([batch_tensor] + hidden_states_tensor)
                else:
                    predictions = self.model(batch_tensor)

                if isinstance(predictions, tuple):
                    predictions, hs = predictions
                    if not return_hs:
                        return_hs = [[] for _ in hs]
                    for ith, ith_layer_hs in enumerate(hs):
                        return_hs[ith].append(ith_layer_hs.numpy())
                else:
                    predictions = predictions.numpy()

                return_scores.append(predictions)

        return_scores = np.concatenate(return_scores, axis=0)

        # Adjust the predicted output to match the number of valid rows in X
        return_scores_truncated = return_scores[:num_valid_rows]

        if hs is not None:
            # Check dimensions and concatenate properly
            if len(return_hs[0][0].shape) == 3:  # (layers, batch_size, hidden_dim)
                return_hs = [np.concatenate(hs_layer, axis=1) for hs_layer in return_hs]
            elif len(return_hs[0][0].shape) == 2:  # (batch_size, hidden_dim)
                return_hs = [np.concatenate(hs_layer, axis=0) for hs_layer in return_hs]
            return return_scores_truncated, return_hs
        else:
            return return_scores_truncated