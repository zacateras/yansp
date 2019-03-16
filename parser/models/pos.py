import keras

class PosModel(keras.Model):
    def __init__(
        self,
        pos_count: int,
        dense_size: int,
        dropout: float,
        *args, **kwargs):

        super(PosModel, self).__init__(*args, **kwargs)

        self.dense_hidden = keras.layers.Dense(
            units=dense_size,
            activation=keras.activations.tanh
        )
        self.dropout_hidden = keras.layers.Dropout(dropout)

        self.dense = keras.layers.Dense(
            units=pos_count,
            activation=keras.activations.tanh
        )
        self.dropout = keras.layers.Dropout(dropout)

        self.softmax = keras.layers.Activation(
            activation=keras.activations.softmax,
            name='pos',
        )

    def call(self, inputs):
        x = self.dense_hidden(inputs)
        x = self.dropout_hidden(x)

        x = self.dense(x)
        x = self.dropout(x)

        x = self.softmax(x)

        return x
