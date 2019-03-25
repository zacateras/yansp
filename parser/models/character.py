import keras

class CharacterModel(keras.Model):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        conv_layers: int,
        conv_size: int,
        *args, **kwargs):

        super(CharacterModel, self).__init__(*args, **kwargs)

        self.char_masking = keras.layers.Masking(mask_value=0)

        class CharacterCharModel(keras.Model):
            def __init__(self, *args, **kwargs):

                super(CharacterCharModel, self).__init__(*args, **kwargs)

                self.embedding = keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=embedding_dim)

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

                self.global_max_pooling = keras.layers.GlobalMaxPooling1D()

            def call(self, inputs):
                x = self.embedding(inputs)
                
                for conv in self.conv:
                    x = conv(x)

                x = self.global_max_pooling(x)

                return x

        self.char_model = keras.layers.TimeDistributed(CharacterCharModel())

    def call(self, inputs):
        x = self.char_masking(inputs)
        x = self.char_model(inputs)

        return x
