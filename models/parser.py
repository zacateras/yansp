import keras
import conll
import models

from models import core
from parser.features import F_FORM, F_FORM_CHAR, F_LEMMA_CHAR, F_UPOS, F_FEATS, F_HEAD, F_DEPREL
from utils import Embeddings, Vocab
from typing import List

class ParserModel(keras.Model):
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

        #
        # inputs
        self.word_model = models.WordModel(
            embeddings=word_embeddings,
            dense_size=args.model_word_dense_size,
            dropout=args.model_dropout
        )

        self.char_masking = keras.layers.Masking(mask_value=0)
        self.char_model = keras.layers.TimeDistributed(
            models.CharacterModel(
                vocab_size=char_vocab.size,
                embedding_dim=args.model_char_embedding_dim,
                conv_layers=args.model_char_conv_layers,
                conv_size=args.model_char_conv_size,
                dense_size=args.model_char_dense_size
            )
        )

        self.concat = keras.layers.Concatenate(axis=-1)

        #
        # core
        if args.model_core_type == 'biLSTM':
            self.core_model = core.biLSTM.Encoder(
                lstm_layers=args.model_core_bilstm_layers,
                lstm_units=args.model_core_bilstm_layer_size,
                lstm_dropout=args.model_core_bilstm_layer_dropout,
                dropout=args.model_core_bilstm_dropout,
                noise=args.model_core_bilstm_noise
            )
        elif args.model_core_type == 'transformer':
            self.core_model = core.transformer.Encoder(
                input_dropout=args.model_core_transformer_input_dropout,
                hidden_size=args.model_core_transformer_hidden_size,
                max_length=args.model_core_transformer_sent_max_length,
                layers=args.model_core_transformer_layers,
                attention_key_dense_size=args.model_core_transformer_attention_key_dense_size,
                attention_query_dense_size=args.model_core_transformer_attention_query_dense_size,
                attention_heads=args.model_core_transformer_attention_heads,
                attention_dropout=args.model_core_transformer_attention_dropout,
                pff_filter_size=args.model_core_transformer_pff_filter_size,
                pff_dropout=args.model_core_transformer_pff_dropout,
                layer_dropout=args.model_core_transformer_layer_dropout
            )
        else:
            raise ValueError('args.model_core_type must be one of (biLSTM, transformer).')

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
            conv_layers=args.model_char_conv_layers,
            conv_size=args.model_char_conv_size,
            dense_size=args.model_lemma_dense_size,
            dropout=args.model_dropout
        )

    def call(self, inputs):

        word_inp = inputs[F_FORM]
        word = self.word_model(word_inp)

        char_inp = inputs[F_FORM_CHAR]
        char = self.char_masking(char_inp)
        char = self.char_model(char)

        x = self.concat([word, char])

        x = self.core_model(x)

        lemma = self.lemma_model(x, char_inp)
        upos = self.upos_model(x)
        feats = self.feats_model(x)
        head = self.head_model(x)
        deprel = self.deprel_model(x, head)

        return {
            F_LEMMA_CHAR: lemma,
            F_UPOS: upos,
            F_FEATS: feats,
            F_HEAD: head,
            F_DEPREL: deprel
        }
