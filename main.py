import argparse
import time
import os

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
    validation = True if args.dev_file is not None else False
    
    log('Loading CoNLL-U training file...')
    conllu_train = conll.load_conllu(args.train_file)
    sents_train = conllu_train.sents

    if validation:
        log('Loading CoNLL-U validation file...')
        conllu_dev = conll.load_conllu(args.dev_file)
        sents_dev = conllu_dev.sents

    log('Loading embeddings file...')
    embeddings = utils.Embeddings.from_file(args.wordvec_file)

    vocabs = conllu_train.vocabs
    vocabs[conll.vocab.WORD] = embeddings.vocab

    y_losses = [
        CategoricalCrossentropyLoss(),
        CategoricalCrossentropyLoss(),
        FeatsLoss(),
        HeadLoss(args.loss_cycle_weight, args.loss_cycle_n, args.batch_size),
        CategoricalCrossentropyLoss(),
    ]

    log('Creating encoder...')
    encoder = FeaturesEncoder(vocabs, args)

    log('Creating train generator...')
    generator_train = LenwiseSentBatchGenerator(sents_train, args.batch_size) \
                      if args.batch_lenwise else \
                      RandomSentBatchGenerator(sents_train, args.batch_size)
    generator_train = map(encoder.encode_batch, generator_train)

    if validation:
        log('Creating dev generator...')
        generator_dev = RandomSentBatchGenerator(sents_dev, args.batch_size)
        generator_dev = map(encoder.encode_batch, generator_dev)

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

        for (batch_i, batch) in ((i, next(generator_train)) for i in range(args.batch_per_epoch)):
            loss_summaries = dict()
            
            with tf.GradientTape() as tape:
                y_pred = model(batch.x)
                y_true = batch.y
                
                loss_total = 0.0
                for (feat, yt, yp, l, weight) in zip(F_Y, y_true.values(), y_pred.values(), y_losses, args.loss_weights):
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
                log('Saving checkpoint...')
                checkpoint.save(checkpoint_path + 'model')

            # log to console
            if batch_i % args.loss_summary_global_step == 0:
                log(loss_summaries)

            # log to file
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.loss_summary_global_step):
                for feat, loss in loss_summaries.items():
                    tf.contrib.summary.scalar(feat, loss)

        if validation:
            log('Validation {}/{}.'.format(epoch_i + 1, args.epochs))
            batch_dev = next(generator_dev)

            file_true = args.save_dir + '/validation/batch_true.conllu'
            sents_dev_true = encoder.decode_batch(batch_dev)
            write_batch_to_conll_file(sents_dev_true, file_true)

            batch_dev.y = model(batch_dev.x)

            file_pred = args.save_dir + '/validation/batch_pred.conllu'
            sents_dev_pred = encoder.decode_batch(batch_dev)
            write_batch_to_conll_file(sents_dev_pred, file_pred)

            # result = conll.evaluate(file_true, file_pred)
            # log(result)

import conll.conll18_ud_eval_proxy as conllp

def write_batch_to_conll_file(batch: Batch, file):

    sents = []
    for sent_i in range(len(batch.sents)):
        sent = batch.sents[sent_i]

        words = list()
        for word_i in range(len(sent)):
            columns = [
                str(sent[word_i].id),                       # 1 id
                sent[word_i].form,                          # 2 form
                batch.y[F_LEMMA_CHAR][sent_i][word_i],      # 3 lemma
                batch.y[F_UPOS][sent_i][word_i],            # 4 upos
                '_',                                        # 5 xpos
                '|'.join(batch.y[F_FEATS][sent_i][word_i]), # 6 feats
                str(batch.y[F_HEAD][sent_i][word_i]),       # 7 head
                batch.y[F_DEPREL][sent_i][word_i],          # 8 deprel
                '_',                                        # 9 deps
                '_'                                         # 10 misc
            ]

            words.append(conllp.UDWord(columns))

        sents.append(conllp.UDSentence(words))

    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w+', encoding='utf-8') as f:
        for sent in sents:
            # foreach word except root
            for word in sent.words[1:]:
                f.write('\t'.join(str(column) for column in word.columns) + '\n')

            f.write('\n')
            

if __name__ == '__main__':
    main()
