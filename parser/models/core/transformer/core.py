import keras
import tensorflow as tf

import numpy as np
import math

from .normalization import LayerNorm
from .layers import EncoderLayer, DecoderLayer

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
                    math.log(float(max_timescale) / float(min_timescale)) /
                    (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
                    np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)


    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 
                    'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([length, channels])

    return tf.convert_to_tensor(signal, dtype='float32')

class Encoder(keras.Model):
    def __init__(
        self,
        input_dropout: float,
        hidden_size: int,
        max_length: int,
        layers: int,
        attention_key_dense_size: int,
        attention_query_dense_size: int,
        attention_heads: int,
        attention_dropout: float,
        pff_layers: int,
        pff_filter_size: int,
        pff_dropout: float,
        layer_dropout: float,
        *args, **kwargs):

        super(Encoder, self).__init__(*args, **kwargs)

        self.input_dropout = keras.layers.Dropout(input_dropout)
        self.embedding_projection = keras.layers.Dense(hidden_size)
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        self.enc_all = [
            EncoderLayer(
                hidden_size,
                attention_key_dense_size,
                attention_query_dense_size,
                attention_heads,
                attention_dropout,
                pff_layers,
                pff_filter_size,
                pff_dropout,
                layer_dropout) \
            for i in range(layers)
        ]

        self.norm = LayerNorm()

    def call(self, inputs):
        x = self.input_dropout(inputs)
        x = self.embedding_projection(x)

        x = x + self.timing_signal[:inputs.shape[1], :]

        for enc in self.enc_all:
            x = enc(x)

        x = self.norm(x)

        return x
