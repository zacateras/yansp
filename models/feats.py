import keras

class FeatsModel(keras.Model):
    def __init__(
        self,
        feats_count: int,
        dense_size: int,
        dropout: float):

        super(FeatsModel, self).__init__()

        self.dense_hidden = keras.layers.Dense(
            units=dense_size,
            activation=keras.activations.tanh
        )
        self.dropout_hidden = keras.layers.Dropout(dropout)

        self.dense = keras.layers.Dense(
            units=feats_count,
            activation=keras.activations.tanh
        )
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs):
        x = self.dense_hidden(inputs)
        x = self.dropout_hidden(x)

        x = self.dense(x)
        x = self.dropout(x)

        return x
