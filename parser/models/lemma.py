import keras
import tensorflow as tf

class LemmaModel(keras.Model):
    def __init__(
        self,
        word_max_length: int,
        char_vocab_size: int,
        char_embedding_dim: int,
        conv_layers: int,
        conv_size: int,
        dense_size: int,
        dropout: float):

        super(LemmaModel, self).__init__()

        self.concat = keras.layers.Concatenate(axis=-1)

        class LemmaCharModel(keras.Model):
            def __init__(self):

                super(LemmaCharModel, self).__init__()

                self.extract_char = keras.layers.Lambda(lambda x: x[:, :word_max_length])
                self.extract_core = keras.layers.Lambda(lambda x: x[:, word_max_length:])
                
                self.embedding = keras.layers.Embedding(
                    input_dim=char_vocab_size,
                    output_dim=char_embedding_dim)

                self.dense = keras.layers.Dense(
                    units=dense_size,
                    activation=keras.activations.tanh
                )
                self.dropout = keras.layers.Dropout(dropout)
                self.repeat = keras.layers.RepeatVector(word_max_length)

                self.concat = keras.layers.Concatenate(axis=-1)

                self.conv = [
                    keras.layers.Conv1D(
                        filters=conv_size,
                        kernel_size=3,
                        strides=1,
                        dilation_rate=2 ** i,
                        activation=keras.activations.relu,
                        padding='same',
                        kernel_regularizer=keras.regularizers.l2(0.000001),
                        bias_regularizer=keras.regularizers.l2(0.000001)
                    )
                    for i in range(conv_layers)
                ]

                self.last_conv = keras.layers.Conv1D(
                    filters=char_vocab_size,
                    kernel_size=1,
                    strides=1,
                    dilation_rate=1,
                    activation=None,
                    padding='same',
                    kernel_regularizer=keras.regularizers.l2(0.000001),
                    bias_regularizer=keras.regularizers.l2(0.000001)
                )

                self.softmax = keras.layers.Activation(
                    activation=keras.activations.softmax
                )

            def call(self, inputs):
                
                char = self.extract_char(inputs)
                char = self.embedding(char)

                core = self.extract_core(inputs)
                core = self.dense(core)
                core = self.dropout(core)
                core = self.repeat(core)
                
                x = self.concat([char, core])
                
                for conv in self.conv:
                    x = conv(x)

                x = self.last_conv(x)
                x = self.softmax(x)

                return x

        self.char_model = keras.layers.TimeDistributed(LemmaCharModel())

    def call(self, inputs_core, inputs_char):

        core = inputs_core
        char = tf.dtypes.cast(inputs_char, tf.float32)

        x = self.concat([char, core])

        return self.char_model(x)
