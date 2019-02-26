import argparse

import conll
import utils
import models

import numpy as np

from parser.features import *
from parser.losses import *

import tensorflow as tf
import keras

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordvec_file', type=str, default=None, help='Input file for word vectors.')
    parser.add_argument('--train_file', type=str, default=None, help='Input CoNLL-U train file.')
    parser.add_argument('--dev_file', type=str, default=None, help='Input CoNLL-U dev file.')
    parser.add_argument('--test_file', type=str, default=None, help='Input CoNLL-U test file.')
    parser.add_argument('--save_dir', type=str, default=None, help='Root dir for saving logs and models.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')

    parser.add_argument('--batch_size', type=int, default=1000, help='Size of batches (in words).')

    parser.add_argument('--model_word_dense_size', type=int, default=100, help='Size of word model output dense layer.')
    parser.add_argument('--model_word_max_length', type=int, default=30, help='Maximum length of words.')
    parser.add_argument('--model_char_embedding_dim', type=int, default=100, help='Dimension of character level embeddings.')
    parser.add_argument('--model_char_conv_layers', type=int, default=3, help='Number of convolution layers in character model.')
    parser.add_argument('--model_char_conv_size', type=int, default=30, help='Size of character model convolution layers.')
    parser.add_argument('--model_char_dense_size', type=int, default=100, help='Size of character model output dense layer.')
    parser.add_argument('--model_lstm_layers', type=int, default=2, help='Numer of LSTM layers in core model.')
    parser.add_argument('--model_lstm_units', type=int, default=256, help='Numer of output units for LSTM layers in core model.')
    parser.add_argument('--model_lstm_dropout', type=int, default=2, help='Dropout rate applied to LSTM layers in core model.')
    parser.add_argument('--model_head_dense_size', type=int, default=100, help='Size of head model hidden dense size.')
    parser.add_argument('--model_deprel_dense_size', type=int, default=100, help='Size of deprel model hidden dense size.')
    parser.add_argument('--model_upos_dense_size', type=int, default=100, help='Size of UPOS model hidden dense size.')
    parser.add_argument('--model_feats_max_length', type=int, default=10, help='Maximum length of features.')
    parser.add_argument('--model_feats_dense_size', type=int, default=100, help='Size of feats model hidden dense size.')
    parser.add_argument('--model_lemma_dense_size', type=int, default=25, help='Size of lemma model hidden dense size.')
    parser.add_argument('--model_dropout', type=float, default=0.25, help='Dropout rate applied on default to dropout layers.')
    parser.add_argument('--model_noise', type=float, default=0.2, help='Noise StdDev applied on default to noise layers.')

    parser.add_argument('--model_loss_cycle_weight', type=float, default=1.0, help='Relative weight of cycle loss.')
    parser.add_argument('--model_loss_cycle_n', type=int, default=3, help='Number of cycles to find.')

    args = parser.parse_args()
    return args

def batches_by_words(sents, batch_size_max):
    batch_size = 0
    batch_buffer = []
    for sent in sents:
        batch_buffer.append(sent)
        batch_size += len(sent)

        if batch_size >= batch_size_max:
            yield batch_buffer
            batch_buffer = []
            batch_size = 0

    if batch_size > 0:
        yield batch_buffer

def generator_from(mapping):
    for item in mapping:
        yield (item[0], item[1])

def main():
    tf.enable_eager_execution()

    args = parse_args()
    conllu_train = conll.load_conllu(args.train_file)
    embeddings = utils.Embeddings.from_file(args.wordvec_file)

    x_feats = [F_FORM, F_FORM_CHAR]
    y_feats = [F_LEMMA_CHAR, F_UPOS, F_FEATS, F_HEAD, F_DEPREL]
    y_losses = [
        CategoricalCrossentropyLoss(),
        CategoricalCrossentropyLoss(),
        FeatsLoss(),
        HeadLoss(args.model_loss_cycle_weight, args.model_loss_cycle_n, args.batch_size),
        CategoricalCrossentropyLoss(),
    ]
    y_losses_weights = [0.2, 0.8, 0.05, 0.05, 0.2]

    sents = conllu_train.sents
    vocabs = conllu_train.vocabs
    vocabs[conll.vocab.WORD] = embeddings.vocab
    encoder = FeaturesEncoder(vocabs, args, x_feats=x_feats, y_feats=y_feats)

    sents_by_length = sorted(sents, key=lambda sent: len(sent))
    batches = batches_by_words(sents_by_length, args.batch_size)
    generator = generator_from(map(encoder.encode_batch, batches))

    parser = models.ParserModel(
        args,
        word_embeddings=embeddings,
        vocabs=conllu_train.vocabs
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.9, epsilon=1e-4)

    loss_history = []
    for (batch, (x, y_true)) in enumerate(generator):
        loss_feats = dict()

        if batch % 10 == 0:
            print(loss_feats)
        with tf.GradientTape() as tape:
            y_pred = parser(x)
            
            loss_total = 0.0
            for (feat, yt, yp, l, weight) in zip(y_feats, y_true, y_pred, y_losses, y_losses_weights):
                loss = l(yt, yp)
                loss_feats[feat] = loss.numpy()
                loss_total += weight * loss

        loss_feats['TOTAL'] = loss_total.numpy()
        
        loss_history.append(loss_feats)

        grads = tape.gradient(loss_total, parser.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, parser.trainable_variables), 
            global_step=tf.train.get_or_create_global_step())


if __name__ == '__main__':
    main()
