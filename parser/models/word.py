import keras
from utils.embeddings import Embeddings

class WordModel(keras.Model):
    def __init__(
        self,
        embeddings: Embeddings,
        dense_size: int,
        dropout: float,
        *args, **kwargs):

        super(WordModel, self).__init__(*args, **kwargs)

        self.embedding = keras.layers.Embedding(
            input_dim=embeddings.size,
            output_dim=embeddings.dim,
            mask_zero=True,
            weights=embeddings.vectors,
            trainable=False
        )

        self.dense = keras.layers.Dense(
            units=dense_size,
            activation=keras.activations.tanh
        )

        self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dense(x)
        x = self.dropout(x)

        return x
