import tensorflow as tf

class PosModel(tf.keras.Model):
    def __init__(
        self,
        pos_count: int,
        dense_size: int,
        dropout: float):

        super(PosModel, self).__init__()

        self.dense_hidden = tf.keras.layers.Dense(
            units=dense_size,
            activation=tf.keras.activations.tanh
        )
        self.dropout_hidden = tf.keras.layers.Dropout(dropout)

        self.dense = tf.keras.layers.Dense(
            units=pos_count,
            activation=tf.keras.activations.tanh
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.softmax = tf.keras.layers.Activation(
            activation=tf.keras.activations.softmax,
            name='pos',
        )

    def call(self, inputs):
        x = self.dense_hidden(inputs)
        x = self.dropout_hidden(x)

        x = self.dense(x)
        x = self.dropout(x)

        x = self.softmax(x)

        return x
