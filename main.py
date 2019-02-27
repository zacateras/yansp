import argparse
import time

import conll
import utils
import models

import numpy as np

from parser.features import *
from parser.losses import *
from parser.generators import *

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

    parser.add_argument('--epochs', type=int, default=20, help='Numer of epochs.')
    parser.add_argument('--batch_per_epoch', type=int, default=50, help='Number of batches per epoch.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Size of batches (in words).')
    parser.add_argument('--batch_lenwise', type=bool, default=False, help='If true, sentences will be sorted and processed in length order')

    parser.add_argument('--checkpoint_global_step', type=int, default=10, help='Checkpointing interval.')

    parser.add_argument('--loss_cycle_weight', type=float, default=1.0, help='Relative weight of cycle loss.')
    parser.add_argument('--loss_cycle_n', type=int, default=3, help='Number of cycles to find.')
    parser.add_argument('--loss_weights', default=[0.2, 0.8, 0.05, 0.05, 0.2], help='Losses weights.')
    parser.add_argument('--loss_summary_global_step', type=int, default=1, help='Summary logs reporting interval.')

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

    args = parser.parse_args()
    return args

def log(message):
    print('{} {}'.format(time.strftime("%Y-%m-%d %H:%M:%S"), message))

def main():
    tf.enable_eager_execution()

    log('Parsing args...')
    args = parse_args()
    log('Loading CoNLL-U file...')
    conllu_train = conll.load_conllu(args.train_file)
    log('Loading embeddings file...')
    embeddings = utils.Embeddings.from_file(args.wordvec_file)

    sents = conllu_train.sents
    vocabs = conllu_train.vocabs
    vocabs[conll.vocab.WORD] = embeddings.vocab

    x_feats = [
        F_FORM,
        F_FORM_CHAR
    ]

    y_feats = [
        F_LEMMA_CHAR,
        F_UPOS,
        F_FEATS,
        F_HEAD,
        F_DEPREL
    ]

    y_losses = [
        CategoricalCrossentropyLoss(),
        CategoricalCrossentropyLoss(),
        FeatsLoss(),
        HeadLoss(args.loss_cycle_weight, args.loss_cycle_n, args.batch_size),
        CategoricalCrossentropyLoss(),
    ]

    log('Creating generator...')
    generator = LenwiseSentBatchGenerator(sents, args.batch_size) \
                if args.batch_lenwise else \
                RandomSentBatchGenerator(sents, args.batch_size)

    encoder = FeaturesEncoder(vocabs, args, x_feats=x_feats, y_feats=y_feats)

    generator = map(encoder.encode_batch, generator)

    log('Creating model & optimizer...')
    model = models.ParserModel(args, word_embeddings=embeddings, vocabs=conllu_train.vocabs)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.9, epsilon=1e-4)

    log('Loading checkpoints...')
    checkpoint_path = args.save_dir + '/checkpoints/'
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, optimizer_step=tf.train.get_or_create_global_step())
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    log('Initializing logger...')
    logger = tf.contrib.summary.create_file_writer(args.save_dir + '/logs/')
    logger.set_as_default()

    log('Starting training ({} epochs, {} batches/epoch)...'.format(args.epochs, args.batch_per_epoch))
    for epoch_i in range(args.epochs):

        log('Epoch {}/{}.'.format(epoch_i + 1, args.epochs))

        for (batch_i, (x, y_true)) in ((i, next(generator)) for i in range(args.batch_per_epoch)):
            loss_summaries = dict()
            
            with tf.GradientTape() as tape:
                y_pred = model(x)
                
                loss_total = 0.0
                for (feat, yt, yp, l, weight) in zip(y_feats, y_true, y_pred, y_losses, args.loss_weights):
                    loss = l(yt, yp)
                    loss_summaries[feat] = loss.numpy()
                    loss_total += weight * loss

            loss_summaries['TOTAL'] = loss_total.numpy()

            grads = tape.gradient(loss_total, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables), 
                global_step=tf.train.get_or_create_global_step())

            # save checkpoint
            if batch_i % args.checkpoint_global_step == 0:
                checkpoint.save(checkpoint_path + 'model')

            # log to console
            if batch_i % args.loss_summary_global_step == 0:
                log(loss_summaries)

            # log to file
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.loss_summary_global_step):
                for feat, loss in loss_summaries.items():
                    tf.contrib.summary.scalar(feat, loss)

if __name__ == '__main__':
    main()
