import keras

class HeadModel(keras.Model):
    def __init__(
        self,
        dense_size: int,
        dropout: float,
        *args, **kwargs):

        super(HeadModel, self).__init__(*args, **kwargs)

        self.dependent_dense = keras.layers.Dense(
            units=dense_size,
            activation=keras.activations.tanh
        )
        self.dependent_dropout = keras.layers.Dropout(dropout)

        self.head_dense = keras.layers.Dense(
            units=dense_size,
            activation=keras.activations.tanh
        )
        self.head_dropout = keras.layers.Dropout(dropout)

        self.dot = keras.layers.Dot(axes=2)
        self.softmax = keras.layers.Activation(
            activation=keras.activations.softmax,
            name='head',
        )

    def call(self, inputs):
        dependent = self.dependent_dense(inputs)
        dependent = self.dependent_dropout(dependent)

        head = self.head_dense(inputs)
        head = self.head_dropout(head)

        x = self.dot([dependent, head])
        x = self.softmax(x)

        return x
