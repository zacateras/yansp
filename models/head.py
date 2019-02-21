import tensorflow as tf

class HeadModel(tf.keras.Model):
    def __init__(
        self,
        head_dense_size: int,
        dropout: float):

        super(HeadModel, self).__init__()

        self.dependent_dense = tf.keras.layers.Dense(
            units=head_dense_size,
            activation=tf.keras.activations.tanh
        )
        self.dependent_dropout = tf.keras.Dropout(dropout)

        self.head_dense = tf.keras.layers.Dense(
            units=head_dense_size,
            activation=tf.keras.activations.tanh
        )
        self.head_dropout = tf.keras.layers.Dropout(dropout)

        self.dot = tf.keras.layers.Dot(axes=2)
        self.softmax = tf.keras.layers.Activation(
            activation=tf.keras.activations.softmax,
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
