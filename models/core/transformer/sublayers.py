import tensorflow as tf
import keras

class MultiHeadAttention(keras.Model):
    def __init__(
        self,
        key_dense_size,
        query_dense_size,
        output_dense_size,
        heads_count,
        dropout):

        super(MultiHeadAttention, self).__init__()

        self.heads_count = heads_count
        self.query_scale = (key_dense_size // heads_count)**-0.5

        self.query_dense = keras.layers.Dense(key_dense_size)
        self.key_dense = keras.layers.Dense(key_dense_size)
        self.value_dense = keras.layers.Dense(query_dense_size)
        self.output_dense = keras.layers.Dense(output_dense_size)

        self.softmax = keras.layers.Softmax()

        self.dropout = keras.layers.Dropout(dropout)

    def call(self, query, key, value):

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        query *= self.query_scale

        # TODO add bias mask

        logit = tf.matmul(query, key, transpose_b=True)

        weight = self.softmax(logit)
        weight = self.dropout(weight)

        context = tf.matmul(weight, value)
        context = self._merge_heads(context)

        output = self.output_dense(context)

        return output

    def _split_heads(self, x):
        """
        Splits tensor by adding an additional dimension of heads_count length
        Input:
            x: a Tensor with shape [..., seq_length, encoding_size]
        Returns:
            A Tensor with shape [..., heads_count, seq_length, encoding_size/heads_count]
        """
        shape = x.shape.as_list()
        shape = shape[:-1] + [self.heads_count, shape[-1] // self.heads_count]
        x = tf.reshape(x, shape)

        perm = list(range(len(shape)))
        perm = perm[:-3] + [perm[-2], perm[-3], perm[-1]]
        x = tf.transpose(x, perm)
        
        return x

    def _merge_heads(self, x):
        """
        Merges -3 tensor dimension of heads_count length
        Input:
            x: a Tensor with shape [..., heads_count, seq_length, encoding_size/heads_count]
        Returns:
            A Tensor with shape [..., seq_length, encoding_size]
        """
        shape = x.shape.as_list()
        perm = list(range(len(shape)))
        perm = perm[:-3] + [perm[-2], perm[-3], perm[-1]]
        x = tf.transpose(x, perm)

        shape = shape[:-3] + [shape[-2], shape[-1] * self.heads_count]
        x = tf.reshape(x, shape)

        return x

class PositionwiseFeedForward(keras.Model):
    def __init__(
        self,
        hidden_dense_size,
        output_dense_size,
        dropout):
        super(PositionwiseFeedForward, self).__init__()

        self.hidden_dense = keras.layers.Dense(hidden_dense_size)
        self.output_dense = keras.layers.Dense(output_dense_size)

        self.relu = keras.layers.ReLU()

        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x):
        x = self.hidden_dense(x)
        x = self.output_dense(x)

        x = self.relu(x)
        
        x = self.dropout(x)

        return x

class LayerNorm(keras.layers.Layer):
    """
    Normalization layer from Hinton et. al.: https://arxiv.org/pdf/1607.06450.pdf
    """
    def __init__(
        self,
        features,
        eps=1e-6,
        gamma_initializer='ones',
        beta_initializer='zeros',
        gamma_regularizer=None,
        beta_regularizer=None):

        super(LayerNorm, self).__init__()
        self.features = features
        self.eps = eps
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-2]

        self.gamma = self.add_weight(
            shape=(input_dim, self.features),
            initializer=self.gamma_initializer,
            name='gamma',
            regularizer=self.gamma_regularizer)

        self.beta = self.add_weight(
            shape=(input_dim, self.features),
            initializer=self.beta_initializer,
            name='beta',
            regularizer=self.beta_regularizer)

        super(LayerNorm, self).build(input_shape)

    def call(self, x):
        mean = keras.backend.mean(x, axis=-1, keepdims=True)
        std = keras.backend.std(x, axis=-1, keepdims=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.features
        return tuple(output_shape)

    def get_config(self):
        config = {
            'features': self.features,
            'eps': self.eps,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.gamma_initializer)
        }
        base_config = super(LayerNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
