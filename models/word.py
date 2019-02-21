import tensorflow as tf
from utils.embeddings import Embeddings

class WordModel(tf.keras.Model):
    def __init__(
        self,
        embeddings: Embeddings,
        dense_size: int,
        dropout: float):

        super(WordModel, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=embeddings.size,
            output_dim=embeddings.dim,
            mask_zero=True,
            weights=embeddings.vectors,
            trainable=False
        )

        self.dense = tf.keras.layers.Dense(
            units=dense_size,
            activation=tf.keras.activations.tanh
        )

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dense(x)
        x = self.dropout(x)

        return x
