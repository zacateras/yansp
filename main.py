import argparse

import conll
import utils
import models

import numpy as np

from parser.features import *

import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordvec_file', type=str, default=None, help='Input file for word vectors.')
    parser.add_argument('--train_file', type=str, default=None, help='Input CoNLL-U train file.')
    parser.add_argument('--dev_file', type=str, default=None, help='Input CoNLL-U dev file.')
    parser.add_argument('--test_file', type=str, default=None, help='Input CoNLL-U test file.')
    parser.add_argument('--save_dir', type=str, default=None, help='Root dir for saving logs and models.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')

    parser.add_argument('--model_word_dense_size', type=int, default=100, help='Size of word model output dense layer.')
    parser.add_argument('--model_word_max_length', type=int, default=30, help='Maximum length of words.')
    parser.add_argument('--model_char_embedding_dim', type=int, default=100, help='Dimension of character level embeddings.')
    parser.add_argument('--model_char_conv_layers', type=int, default=3, help='Number of convolution layers in character model.')
    parser.add_argument('--model_char_conv_size', type=int, default=30, help='Size of character model convolution layers.')
    parser.add_argument('--model_char_dense_size', type=int, default=100, help='Size of character model output dense layer.')
    parser.add_argument('--model_lstm_layers', type=int, default=2, help='Numer of LSTM layers in core model.')
    parser.add_argument('--model_lstm_units', type=int, default=512, help='Numer of output units for LSTM layers in core model.')
    parser.add_argument('--model_lstm_dropout', type=int, default=2, help='Dropout rate applied to LSTM layers in core model.')
    parser.add_argument('--model_head_dense_size', type=int, default=100, help='Size of head model hidden dense size.')
    parser.add_argument('--model_deprel_dense_size', type=int, default=100, help='Size of deprel model hidden dense size.')
    parser.add_argument('--model_upos_dense_size', type=int, default=100, help='Size of UPOS model hidden dense size.')
    parser.add_argument('--model_feats_max_length', type=int, default=10, help='Maximum length of features.')
    parser.add_argument('--model_feats_dense_size', type=int, default=100, help='Size of feats model hidden dense size.')
    parser.add_argument('--model_lemma_dense_size', type=int, default=100, help='Size of lemma model hidden dense size.')
    parser.add_argument('--model_dropout', type=float, default=0.25, help='Dropout rate applied on default to dropout layers.')
    parser.add_argument('--model_noise', type=float, default=0.2, help='Noise StdDev applied on default to noise layers.')

    args = parser.parse_args()
    return args

# def cycle_loss(self, y_true, y_pred):
#     loss = 0.0
#     if self.params.cycle_loss_n == 0:
#         return loss

#     yn = y_pred[:, 1:, 1:]
#     for i in range(self.params.cycle_loss_n):
#         loss += K.sum(tf.trace(yn))/self.params.batch_size
#         yn = K.batch_dot(yn, y_pred[:, 1:, 1:])

#     return loss

# def head_loss(self, y_true, y_pred):
#     loss = 0.0
#     loss += categorical_crossentropy(y_true, y_pred)
#     loss += self.params.cycle_loss_weight*self.cycle_loss(y_true, y_pred)

#     return loss

# def lemma_loss(self, y_true, y_pred):
#     y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#     return -K.mean(K.sum(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred), axis=-1)) 

# def feats_loss(self, y_true, y_pred):
#     loss = 0.0
#     slices = self.targets_factory.encoders['feats'].slices
#     for cat, (min_idx, max_idx) in slices.items():
#         y_pred_cat = Activation('softmax')(y_pred[:, :, min_idx:max_idx])
#         y_true_cat = y_true[:, :, min_idx:max_idx]
#         loss += categorical_crossentropy(y_true_cat, y_pred_cat)

#     return loss

def main():
    args = parse_args()
    conllu_train = conll.load_conllu(args.train_file)
    embeddings = utils.Embeddings.from_file(args.wordvec_file)

    vocabs = conllu_train.vocabs
    vocabs[conll.vocab.WORD] = embeddings.vocab

    encoder = FeaturesEncoder(vocabs, args)
    batch_sents = [next(conllu_train.sents) for _ in range(32)]
    batch = encoder.encode_batch(batch_sents)

    parser = models.ParserModel(
        args,
        word_embeddings=embeddings,
        vocabs=conllu_train.vocabs
    )

    parser.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001, clipvalue=5.0, beta_1=0.9, beta_2=0.9, decay=1e-4)
    )

    output = parser(batch[F_FORM], batch[F_FORM_CHAR])

    pass

if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
