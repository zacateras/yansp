import keras

class DeprelModel(keras.Model):
    def __init__(
        self,
        deprel_count: int,
        dense_size: int,
        dropout: float,
        *args, **kwargs):

        super(DeprelModel, self).__init__(*args, **kwargs)

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

        self.permute = keras.layers.Lambda(lambda x: keras.backend.permute_dimensions(x, (0, 2, 1))) # transpose?

        self.dot = keras.layers.Dot(axes=2)
        self.concat = keras.layers.Concatenate(axis=2)
        self.dense = keras.layers.Dense(units=deprel_count)
        self.dropout = keras.layers.Dropout(dropout)

        self.softmax = keras.layers.Activation(
            activation=keras.activations.softmax
        )

    def call(self, inputs_core, inputs_head):
        dependent = self.dependent_dense(inputs_core)
        dependent = self.dependent_dropout(dependent)

        head = self.head_dense(inputs_core)
        head = self.head_dropout(head)
        head = self.permute(head)
        
        x = self.dot([inputs_head, head])
        x = self.concat([x, dependent])
        x = self.dense(x)
        x = self.dropout(x)
        x = self.softmax(x)

        return x
