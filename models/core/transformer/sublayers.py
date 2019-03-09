import tensorflow as tf
import keras

from .normalization import LayerNorm

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
        hidden_dense_size=2048,
        output_dense_size=512,
        dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.hidden_dense = keras.layers.Dense(hidden_dense_size)
        self.output_dense = keras.layers.Dense(output_dense_size)
        self.relu = keras.layers.ReLU()
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, x):
        x = self.hidden_dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_dense(x)

        return x

class AddAndNorm(keras.Model):
    def __init__(self, dropout: float):
        super(AddAndNorm, self).__init__()

        self.dropout = keras.layers.Dropout(dropout)
        self.norm = LayerNorm()

    def call(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
