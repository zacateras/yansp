import keras
import conll

from .word import WordModel
from .character import CharacterModel
from .core import biLSTM, transformer
from .lemma import LemmaModel
from .pos import PosModel
from .feats import FeatsModel
from .head import HeadModel
from .deprel import DeprelModel

import parser.features as F

from utils import Embeddings, Vocab, count_variables
from typing import List

class ParserModel(keras.Model):
    def __init__(
        self,
        params,
        word_embeddings: Embeddings,
        vocabs: List[Vocab],
        *args, **kwargs):

        super(ParserModel, self).__init__(*args, **kwargs)

        char_vocab = vocabs[conll.vocab.CHAR]
        deprel_vocab = vocabs[conll.vocab.DEPREL]
        upos_vocab = vocabs[conll.vocab.UPOS]
        feats_vocab = vocabs[conll.vocab.FEATS]

        #
        # inputs
        self.word_model = WordModel(
            embeddings=word_embeddings,
            dense_size=params.model_word_dense_size,
            dropout=params.model_dropout,
            name='word'
        )

        self.char_model = CharacterModel(
            vocab_size=char_vocab.size,
            embedding_dim=params.model_char_embedding_dim,
            conv_layers=params.model_char_conv_layers,
            conv_size=params.model_char_conv_size,
            dense_size=params.model_char_dense_size,
            name='char'
        )

        self.concat = keras.layers.Concatenate(axis=-1)

        #
        # core
        if params.model_core_type == 'biLSTM':
            self.core_model = biLSTM.Encoder(
                lstm_layers=params.model_core_bilstm_layers,
                lstm_units=params.model_core_bilstm_layer_size,
                lstm_dropout=params.model_core_bilstm_layer_dropout,
                dropout=params.model_core_bilstm_dropout,
                noise=params.model_core_bilstm_noise,
                name='core'
            )
        elif params.model_core_type == 'transformer':
            self.core_model = transformer.Encoder(
                input_dropout=params.model_core_transformer_input_dropout,
                hidden_size=params.model_core_transformer_hidden_size,
                max_length=params.model_core_transformer_sent_max_length,
                layers=params.model_core_transformer_layers,
                attention_key_dense_size=params.model_core_transformer_attention_key_dense_size,
                attention_query_dense_size=params.model_core_transformer_attention_query_dense_size,
                attention_heads=params.model_core_transformer_attention_heads,
                attention_dropout=params.model_core_transformer_attention_dropout,
                pff_filter_size=params.model_core_transformer_pff_filter_size,
                pff_dropout=params.model_core_transformer_pff_dropout,
                layer_dropout=params.model_core_transformer_layer_dropout,
                name='core'
            )
        else:
            raise ValueError('params.model_core_type must be one of (biLSTM, transformer).')

        #
        # outputs
        self.lemma_model = LemmaModel(
            word_max_length=params.model_word_max_length,
            char_vocab_size=char_vocab.size,
            char_embedding_dim=params.model_char_embedding_dim,
            conv_layers=params.model_char_conv_layers,
            conv_size=params.model_char_conv_size,
            dense_size=params.model_lemma_dense_size,
            dropout=params.model_dropout,
            name='lemma'
        )

        self.upos_model = PosModel(
            pos_count=upos_vocab.size,
            dense_size=params.model_upos_dense_size,
            dropout=params.model_dropout,
            name='pos'
        )

        self.head_model = HeadModel(
            dense_size=params.model_head_dense_size,
            dropout=params.model_dropout,
            name='head'
        )

        self.feats_model = FeatsModel(
            feats_count=feats_vocab.size,
            dense_size=params.model_feats_dense_size,
            dropout=params.model_dropout,
            name='feats'
        )

        self.deprel_model = DeprelModel(
            deprel_count=deprel_vocab.size,
            dense_size=params.model_deprel_dense_size,
            dropout=params.model_dropout,
            name='deprel'
        )

    def call(self, inputs):

        word_inp = inputs[F.FORM]
        word = self.word_model(word_inp)

        char_inp = inputs[F.FORM_CHAR]
        char = self.char_model(char_inp)

        x = self.concat([word, char])

        x = self.core_model(x)

        lemma = self.lemma_model(x, char_inp)
        upos = self.upos_model(x)
        feats = self.feats_model(x)
        head = self.head_model(x)
        deprel = self.deprel_model(x, head)

        return {
            F.LEMMA_CHAR: lemma,
            F.UPOS: upos,
            F.FEATS: feats,
            F.HEAD: head,
            F.DEPREL: deprel
        }
