import keras

class CharacterModel(keras.Model):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        conv_layers: int,
        conv_size: int,
        dense_size: int):

        super(CharacterModel, self).__init__()

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

        self.dense = keras.layers.Dense(dense_size)

    def call(self, inputs):
        x = self.embedding(inputs)
        
        for conv in self.conv:
            x = conv(x)

        x = self.global_max_pooling(x)

        return x
