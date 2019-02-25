import tensorflow as tf
import conll
import models
from utils import Embeddings, Vocab
from typing import List

class ParserModel(tf.keras.Model):
    def __init__(
        self,
        args,
        word_embeddings: Embeddings,
        vocabs: List[Vocab]):

        super(ParserModel, self).__init__()

        char_vocab = vocabs[conll.vocab.CHAR]
        deprel_vocab = vocabs[conll.vocab.DEPREL]
        upos_vocab = vocabs[conll.vocab.UPOS]
        feats_vocab = vocabs[conll.vocab.FEATS]
        lemma_vocab = vocabs[conll.vocab.LEMMA]

        #
        # inputs
        self.word_model = models.WordModel(
            embeddings=word_embeddings,
            dense_size=args.model_word_dense_size,
            dropout=args.model_dropout
        )

        self.char_masking = tf.keras.layers.Masking(mask_value=0)
        self.char_model = tf.keras.layers.TimeDistributed(
            models.CharacterModel(
                vocab_size=char_vocab.size,
                embedding_dim=args.model_char_embedding_dim,
                conv_layers=args.model_char_conv_layers,
                conv_size=args.model_char_conv_size,
                dense_size=args.model_char_dense_size
            )
        )

        self.concat = tf.keras.layers.Concatenate(axis=-1)

        #
        # core
        self.core_model = models.CoreModel(
            lstm_layers=args.model_lstm_layers,
            lstm_units=args.model_lstm_units,
            lstm_dropout=args.model_lstm_dropout,
            dropout=args.model_dropout,
            noise=args.model_noise
        )

        #
        # outputs
        self.head_model = models.HeadModel(
            dense_size=args.model_head_dense_size,
            dropout=args.model_dropout
        )

        self.deprel_model = models.DeprelModel(
            deprel_count=deprel_vocab.size,
            dense_size=args.model_deprel_dense_size,
            dropout=args.model_dropout
        )

        self.upos_model = models.PosModel(
            pos_count=upos_vocab.size,
            dense_size=args.model_upos_dense_size,
            dropout=args.model_dropout
        )

        self.feats_model = models.FeatsModel(
            feats_count=feats_vocab.size,
            dense_size=args.model_feats_dense_size,
            dropout=args.model_dropout
        )

        self.lemma_model = models.LemmaModel(
                word_max_length=args.model_word_max_length,
                char_vocab_size=char_vocab.size,
                char_embedding_dim=args.model_char_embedding_dim,
                lemma_vocab_size=lemma_vocab.size,
                conv_layers=args.model_char_conv_layers,
                conv_size=args.model_char_conv_size,
                dense_size=args.model_lemma_dense_size,
                dropout=args.model_dropout
            )

    def call(self, inputs_word, inputs_char):

        word = self.word_model(inputs_word)
        char = self.char_masking(inputs_char)
        char = self.char_model(char)

        x = self.concat([word, char])

        x = self.core_model(x)

        lemma = self.lemma_model(x, inputs_char)
        upos = self.upos_model(x)
        feats = self.feats_model(x)
        head = self.head_model(x)
        deprel = self.deprel_model(x, head)

        return [lemma, upos, feats, head, deprel]
