import numpy as np
import tensorflow as tf

Dense = tf.keras.layers.Dense
Multiply = tf.keras.layers.Multiply
LayerNorm = tf.keras.layers.LayerNormalization
Activation = tf.keras.layers.Activation
Add = tf.keras.layers.Add

tf.config.run_functions_eagerly(False)


# Layer utility functions.
def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True):
    """Returns simple Keras linear layer.
      Args:
        size: Output size
        activation: Activation function to apply if required
        use_time_distributed: Whether to apply layer across time
        use_bias: Whether bias should be included in layer
    """
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear


def add_and_norm(x_list):
    """Applies skip connection followed by layer normalisation.

  Args:
    x_list: List of inputs to sum for skip connection

  Returns:
    Tensor output from layer.
  """
    tmp = Add()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp

def apply_gating_layer(x,
                       hidden_layer_size,
                       dropout_rate=None,
                       use_time_distributed=True,
                       activation=None):
    """Applies a Gated Linear Unit (GLU) to an input.

  Args:
    x: Input to gating layer
    hidden_layer_size: Dimension of GLU
    dropout_rate: Dropout rate to apply if any
    use_time_distributed: Whether to apply across time
    activation: Activation function to apply to the linear feature transform if
      necessary

  Returns:
    Tuple of tensors for: (GLU output, gate)
  """

    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation=activation))(
            x)
        gated_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(
            x)
    else:
        activation_layer = tf.keras.layers.Dense(
            hidden_layer_size, activation=activation)(
            x)
        gated_layer = tf.keras.layers.Dense(
            hidden_layer_size, activation='sigmoid')(
            x)

    return tf.keras.layers.Multiply()([activation_layer,
                                       gated_layer]), gated_layer


def gated_residual_network(x,
                           hidden_layer_size,
                           output_size=None,
                           dropout_rate=None,
                           use_time_distributed=True,
                           additional_context=None,
                           return_gate=False):
    """Applies the gated residual network (GRN) as defined in paper.

      Args:
        x: Network inputs
        hidden_layer_size: Internal state size
        output_size: Size of output layer
        dropout_rate: Dropout rate if dropout is applied
        use_time_distributed: Whether to apply network across time dimension
        additional_context: Additional context vector to use if relevant
        return_gate: Whether to return GLU gate for diagnostic purposes

      Returns:
        Tuple of tensors for: (GRN output, GLU gate)
    """
    # Setup skip connection
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        skip = linear(x)

    # Apply feedforward network
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
        x)
    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False)(
            additional_context)
    hidden = tf.keras.layers.Activation('elu')(hidden)
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
        hidden)

    gating_layer, gate = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None)

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])

    
def hadamard_attention(input_layer, hidden_layer_size, dropout):
    """Apply temporal variable selection networks.
        Returns:
        Processed tensor outputs.
    """
    
    
    _, time_steps, num_inputs = input_layer.get_shape().as_list()

    # Variable selection weights
    mlp_outputs, static_gate = gated_residual_network(
        input_layer,
        hidden_layer_size,
        output_size=num_inputs,
        dropout_rate=dropout,
        use_time_distributed=True,
        additional_context=None,
        return_gate=True)
    sparse_weights = tf.keras.layers.Activation('softmax', name="softmax_dyn")(mlp_outputs)

    # Score multiplication
    combined = tf.keras.layers.Multiply()([sparse_weights, input_layer])

    return combined, sparse_weights

    
