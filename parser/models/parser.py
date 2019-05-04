import keras
import tensorflow as tf
import conll
import numpy as np

from .word import WordModel
from .character import CharacterModel
from .core import biLSTM, transformer
from .lemma import LemmaModel
from .pos import PosModel
from .feats import FeatsModel
from .head import HeadModel
from .deprel import DeprelModel

import parser.features as F

from utils import Embeddings, Vocab, count_variables, log
from typing import List

class ParserModel(keras.Model):
    def __init__(
        self,
        params,
        vocabs,
        has_checkpoint=False,
        *args, **kwargs):

        super(ParserModel, self).__init__(*args, **kwargs)

        #
        # inputs
        if 'word' in params['model_inputs']:
            if not has_checkpoint:
                log('Loading embeddings file...')
                embeddings = Embeddings.from_file(params['wordvec_file'])
                vectors = embeddings.vectors
                vocabs[conll.vocab.WORD] = embeddings.vocab
            else:
                embeddings = Embeddings.from_file(params['wordvec_file'], load_vectors=False)
                vectors = np.zeros((embeddings.size, embeddings.dim))

            self.word_model = WordModel(
                embeddings_weights=vectors,
                embeddings_size=embeddings.size,
                embeddings_dim=embeddings.dim,
                dense_size=params['model_word_dense_size'],
                dropout=params['model_dropout'],
                name='word'
            )

        if 'char' in params['model_inputs']:
            self.char_model = CharacterModel(
                vocab_size=vocabs[conll.vocab.CHAR].size,
                embedding_dim=params['model_char_embedding_dim'],
                conv_layers=params['model_char_conv_layers'],
                conv_size=params['model_char_conv_size'],
                name='char'
            )

        self.concat = keras.layers.Concatenate(axis=-1)

        #
        # core
        if params['model_core_type'] == 'biLSTM':
            self.core_model = biLSTM.Encoder(
                lstm_layers=params['model_core_bilstm_layers'],
                lstm_units=params['model_core_bilstm_layer_size'],
                lstm_dropout=params['model_core_bilstm_layer_dropout'],
                dropout=params['model_core_bilstm_dropout'],
                noise=params['model_core_bilstm_noise'],
                name='core'
            )
        elif params['model_core_type'] == 'transformer':
            self.core_model = transformer.Encoder(
                input_dropout=params['model_core_transformer_input_dropout'],
                use_timing_signal=params['model_core_transformer_use_timing_signal'],
                hidden_size=params['model_core_transformer_hidden_size'],
                max_length=params['model_core_transformer_sent_max_length'],
                layers=params['model_core_transformer_layers'],
                layers_direction=params['model_core_transformer_layers_direction'],
                attention_key_dense_size=params['model_core_transformer_attention_key_dense_size'],
                attention_value_dense_size=params['model_core_transformer_attention_value_dense_size'],
                attention_heads=params['model_core_transformer_attention_heads'],
                attention_dropout=params['model_core_transformer_attention_dropout'],
                pff_layers=params['model_core_transformer_pff_layers'],
                pff_filter_size=params['model_core_transformer_pff_filter_size'],
                pff_dropout=params['model_core_transformer_pff_dropout'],
                layer_dropout=params['model_core_transformer_layer_dropout'],
                name='core'
            )
        else:
            raise ValueError('params[\'model_core_type\'] must be one of (biLSTM, transformer).')

        #
        # outputs
        self.lemma_model = LemmaModel(
            word_max_length=params['model_word_max_length'],
            char_vocab_size=vocabs[conll.vocab.CHAR].size,
            char_embedding_dim=params['model_char_embedding_dim'],
            conv_layers=params['model_char_conv_layers'],
            conv_size=params['model_char_conv_size'],
            dense_size=params['model_lemma_dense_size'],
            dropout=params['model_dropout'],
            name='lemma'
        )

        self.upos_model = PosModel(
            pos_count=vocabs[conll.vocab.UPOS].size,
            dense_size=params['model_upos_dense_size'],
            dropout=params['model_dropout'],
            name='pos'
        )

        self.head_model = HeadModel(
            dense_size=params['model_head_dense_size'],
            dropout=params['model_dropout'],
            name='head'
        )

        self.feats_model = FeatsModel(
            feats_count=vocabs[conll.vocab.FEATS].size,
            dense_size=params['model_feats_dense_size'],
            dropout=params['model_dropout'],
            name='feats'
        )

        self.deprel_model = DeprelModel(
            deprel_count=vocabs[conll.vocab.DEPREL].size,
            dense_size=params['model_deprel_dense_size'],
            dropout=params['model_dropout'],
            name='deprel'
        )

    def call(self, inputs):

        x = []

        if hasattr(self, 'word_model'):
            word_inp = tf.dtypes.cast(inputs[F.FORM], tf.float32)
            word = self.word_model(word_inp)
            x.append(word)
        
        char_inp = tf.dtypes.cast(inputs[F.FORM_CHAR], tf.float32)
        
        if hasattr(self, 'char_model'):
            char = self.char_model(char_inp)
            x.append(char)

        x = self.concat(x) if len(x) > 1 else x[0]

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
