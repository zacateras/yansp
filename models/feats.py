import tensorflow as tf

class FeatsModel(tf.keras.Model):
    def __init__(
        self,
        feats_count: int,
        hidden_dense_size: int,
        dropout: float):

        super(FeatsModel, self).__init__()

        self.dense_hidden = tf.keras.layers.Dense(
            units=head_dense_size,
            activation=tf.keras.activations.tanh
        )
        self.dropout_hidden = tf.keras.Dropout(dropout)

        self.dense = tf.keras.layers.Dense(
            units=feats_count,
            activation=tf.keras.activations.tanh
        )
        self.dropout = tf.keras.layers.Dropout(dropout, name='feats')

    def call(self, inputs):
        x = self.dense_hidden(inputs)
        x = self.dense_dropout(x)

        x = self.dense(x)
        x = self.dropout(x)

        return x
