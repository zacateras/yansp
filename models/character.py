import tensorflow as tf

class CharacterModel(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        char_emb_size: int,
        conv_lyrs: int,
        dense_size: int):

        super(CharacterModel, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=char_emb_size)

        self.conv = [
            tf.keras.layers.Conv1D(
                filters=char_emb_size,
                activation=tf.nn.leaky_relu
            )
            for _ in range(conv_lyrs)
        ]

        self.dense = tf.keras.layers.Dense(dense_size)

    def call(self, inputs):
        x = self.embedding(inputs)
        
        for conv in self.conv:
            x = conv(x)

        return self.dense(x)
