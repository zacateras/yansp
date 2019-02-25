import tensorflow as tf

class DeprelModel(tf.keras.Model):
    def __init__(
        self,
        deprel_count: int,
        dense_size: int,
        dropout: float):

        super(DeprelModel, self).__init__()

        self.dependent_dense = tf.keras.layers.Dense(
            units=dense_size,
            activation=tf.keras.activations.tanh
        )
        self.dependent_dropout = tf.keras.layers.Dropout(dropout)

        self.head_dense = tf.keras.layers.Dense(
            units=dense_size,
            activation=tf.keras.activations.tanh
        )
        self.head_dropout = tf.keras.layers.Dropout(dropout)

        self.permute = tf.keras.layers.Lambda(lambda x: tf.keras.backend.permute_dimensions(x, (0, 2, 1))) # transpose?

        self.dot = tf.keras.layers.Dot(axes=2)
        self.concat = tf.keras.layers.Concatenate(axis=2)
        self.dense = tf.keras.layers.Dense(units=deprel_count)
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.softmax = tf.keras.layers.Activation(
            activation=tf.keras.activations.softmax,
            name='deprel'
        )

    def call(self, inputs_core, inputs_head):
        dependent = self.dependent_dense(inputs_core)
        dependent = self.dependent_dropout(dependent)

        head = self.head_dense(inputs_core)
        head = self.head_dropout(head)
        head = self.permute(head)
        
        x = self.dot([inputs_head, head])
        x = self.concat([x, dependent])
        x = self.dropout(x)
        x = self.softmax(x)

        return x
