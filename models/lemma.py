import tensorflow as tf

class LemmaModel(tf.keras.Model):
    def __init__(
        self,
        word_max_length: int,
        char_vocab_size: int,
        char_embedding_dim: int,
        lemma_vocab_size: int,
        conv_layers: int,
        conv_size: int,
        dense_size: int,
        dropout: float):

        super(LemmaModel, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=char_vocab_size,
            output_dim=char_embedding_dim)

        self.dense = tf.keras.layers.Dense(
            units=dense_size,
            activation=tf.keras.activations.tanh
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.repeat = tf.keras.layers.RepeatVector(word_max_length)

        self.concat = tf.keras.layers.Concatenate(axis=2)

        self.conv = [
            tf.keras.layers.Conv1D(
                filters=conv_size,
                kernel_size=3,
                strides=1,
                dilation_rate=2 ^ i,
                activation=tf.keras.activations.relu,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(0.000001),
                bias_regularizer=tf.keras.regularizers.l2(0.000001)
            )
            for i in range(conv_layers)
        ]

        self.last_conv = tf.keras.layers.Conv1D(
            filters=lemma_vocab_size,
            kernel_size=1,
            strides=1,
            dilation_rate=1,
            activation=None,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.000001),
            bias_regularizer=tf.keras.regularizers.l2(0.000001)
        )

        self.softmax = tf.keras.layers.Activation(
            activation=tf.keras.activations.softmax
        )

    def call(self, inputs_core, inputs_char):
        char = self.embedding(inputs_char)

        core = self.dense(inputs_core)
        core = self.droput(core)
        core = self.repeat(core)
        
        x = self.concat([char, core])
        
        for conv in self.conv:
            x = conv(x)

        x = self.last_conv(x)
        x = self.softmax(x)

        return x
