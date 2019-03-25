import keras

from .sublayers import MultiHeadAttention, PositionwiseFeedForward, AddAndNorm
from .normalization import LayerNorm

class EncoderLayer(keras.Model):
    def __init__(
        self,
        hidden_size,
        attention_key_dense_size,
        attention_value_dense_size,
        attention_heads_count,
        attention_dropout,
        pff_layers,
        pff_filter_size,
        pff_dropout,
        layer_dropout,
        *args, **kwargs):

        super(EncoderLayer, self).__init__(*args, **kwargs)

        self.mha = MultiHeadAttention(
            key_dense_size=attention_key_dense_size,
            value_dense_size=attention_value_dense_size,
            output_dense_size=hidden_size,
            heads_count=attention_heads_count,
            dropout=attention_dropout)

        self.mha_connection = AddAndNorm(layer_dropout)

        self.pff = PositionwiseFeedForward(
            hidden_layers=pff_layers,
            hidden_dense_size=pff_filter_size,
            output_dense_size=hidden_size,
            dropout=pff_dropout)

        self.pff_connection = AddAndNorm(layer_dropout)

    def call(self, inputs):
        x = inputs
        
        x = self.mha_connection(x, lambda x: self.mha(x, x, x))
        x = self.pff_connection(x, lambda x: self.pff(x))

        return x

class DecoderLayer(keras.Model):
    def __init__(
        self,
        hidden_size,
        attention_key_dense_size,
        attention_query_dense_size,
        attention_heads_count,
        attention_dropout,
        pff_filter_size,
        pff_dropout,
        layer_dropout):

        super(DecoderLayer, self).__init__()

        self.mha_dec = MultiHeadAttention(
            key_dense_size=attention_key_dense_size,
            value_dense_size=attention_query_dense_size,
            output_dense_size=hidden_size,
            heads_count=attention_heads_count,
            dropout=attention_dropout)

        self.mha_dec_connection = AddAndNorm(layer_dropout)

        self.mha_enc_dec = MultiHeadAttention(
            key_dense_size=attention_key_dense_size,
            value_dense_size=attention_query_dense_size,
            output_dense_size=hidden_size,
            heads_count=attention_heads_count,
            dropout=attention_dropout)

        self.mha_enc_dec_connection = AddAndNorm(layer_dropout)

        self.pff = PositionwiseFeedForward(
            hidden_dense_size=pff_filter_size,
            output_dense_size=hidden_size,
            dropout=pff_dropout)

        self.pff_connection = AddAndNorm(layer_dropout)

    def call(self, inputs):

        x, encoder_outputs = inputs

        x = self.mha_dec_connection(x, lambda x: self.mha_dec(x, x, x))
        x = self.mha_enc_dec_connection(x, lambda x: self.mha_enc_dec(x, encoder_outputs, encoder_outputs))
        x = self.pff_connection(x, self.pff)

        return x
