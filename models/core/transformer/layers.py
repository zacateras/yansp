import keras

from .sublayers import MultiHeadAttention, PositionwiseFeedForward, LayerNorm

class EncoderLayer(keras.Model):
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

        super(EncoderLayer, self).__init__()

        self.mha_norm = LayerNorm(hidden_size)
        self.mha = MultiHeadAttention(
            attention_key_dense_size,
            attention_query_dense_size,
            hidden_size,
            attention_heads_count,
            attention_dropout)

        self.pff_norm = LayerNorm(hidden_size)
        self.pff = PositionwiseFeedForward(
            hidden_dense_size=pff_filter_size,
            output_dense_size=hidden_size,
            dropout=pff_dropout)

        self.dropout = keras.layers.Dropout(layer_dropout)

    def call(self, inputs):
        x = inputs

        # x_norm = self.mha_norm(x)
        y = self.mha(x, x, x)

        x = self.dropout(x + y)

        # x_norm = self.pff_norm(x)
        y = self.pff(x)

        y = self.dropout(x + y)

        return y

class DecoderLayer(keras.Model):
    def __init__(
        self,
        hidden_size,
        attention_key_dense_units,
        attention_query_dense_units,
        attention_heads_count,
        attention_dropout,
        pff_filter_size,
        pff_dropout,
        layer_dropout):

        super(DecoderLayer, self).__init__()

        self.mha_dec_norm = LayerNorm(hidden_size)
        self.mha_dec = MultiHeadAttention(
            attention_key_dense_units,
            attention_query_dense_units,
            hidden_size,
            attention_heads_count,
            attention_dropout)

        self.mha_enc_dec_norm = LayerNorm(hidden_size)
        self.mha_enc_dec = MultiHeadAttention(
            attention_key_dense_units,
            attention_query_dense_units,
            hidden_size,
            attention_heads_count,
            attention_dropout)

        self.pff_norm = LayerNorm(hidden_size)
        self.pff = PositionwiseFeedForward(
            hidden_dense_size=pff_filter_size,
            output_dense_size=hidden_size,
            dropout=pff_dropout)

        self.dropout = keras.layers.Dropout(layer_dropout)

    def call(self, inputs):

        x, encoder_outputs = inputs

        x_norm = self.mha_dec_norm(x)
        y = self.mha_dec(x_norm, x_norm, x_norm)
        
        x = self.dropout(x + y)

        x_norm = self.mha_enc_dec_norm(x)
        y = self.mha_enc_dec(x_norm, encoder_outputs, encoder_outputs)
        
        x = self.dropout(x + y)
        
        x_norm = self.pff_norm(x)        
        y = self.pff(x_norm)
        
        y = self.dropout(x + y)
        
        return y, encoder_outputs
