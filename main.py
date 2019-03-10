import argparse
import time

import tensorflow as tf
import numpy as np

import conll
import utils

from utils.generators import LenwiseBatchGenerator, RandomBatchGenerator, AllAtOnceBatchGenerator
from parser.encoders import FeaturesEncoder
from parser.models import ParserModel
import parser.losses
import parser.scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordvec_file', type=str, default=None, help='Input file for word vectors.')
    parser.add_argument('--train_file', type=str, default=None, help='Input CoNLL-U train file.')
    parser.add_argument('--dev_file', type=str, default=None, help='Input CoNLL-U dev file.')
    parser.add_argument('--test_file', type=str, default=None, help='Input CoNLL-U test file.')
    parser.add_argument('--save_dir', type=str, default=None, help='Root dir for saving logs and models.')

    parser.add_argument('--mode', default='train', type=str, choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')

    parser.add_argument('--epochs', type=int, default=20, help='Numer of epochs.')
    parser.add_argument('--epochs_early_stopping', type=int, default=5, help='Number of epochs w/o loss decrease required for early stopping.')
    parser.add_argument('--batch_per_epoch', type=int, default=50, help='Number of batches per epoch.')
    parser.add_argument('--batch_per_summary', type=int, default=1, help='Summary logs reporting interval.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Size of batches (in words).')
    parser.add_argument('--batch_size_dev', type=int, default=None, help='Size of batches (in words) during validation phase. If None then whole file is used.')
    parser.add_argument('--batch_lenwise', type=bool, default=False, help='If true, sentences will be sorted and processed in length order')

    parser.add_argument('--loss_cycle_weight', type=float, default=1.0, help='Relative weight of cycle loss.')
    parser.add_argument('--loss_cycle_n', type=int, default=3, help='Number of cycles to find.')
    parser.add_argument('--loss_weights', default=[0.2, 0.8, 0.05, 0.05, 0.2], help='Losses weights.')


    parser.add_argument('--model_word_dense_size', type=int, default=100, help='Size of word model output dense layer.')
    parser.add_argument('--model_word_max_length', type=int, default=30, help='Maximum length of words.')
    parser.add_argument('--model_char_embedding_dim', type=int, default=100, help='Dimension of character level embeddings.')
    parser.add_argument('--model_char_conv_layers', type=int, default=3, help='Number of convolution layers in character model.')
    parser.add_argument('--model_char_conv_size', type=int, default=30, help='Size of character model convolution layers.')
    parser.add_argument('--model_char_dense_size', type=int, default=100, help='Size of character model output dense layer.')

    parser.add_argument('--model_core_type', default='transformer', type=str, choices=['transformer', 'biLSTM'], help='Type of core model used (either transformer or biLSTM).')
    parser.add_argument('--model_core_bilstm_layers', type=int, default=2, help='Numer of LSTM layers in biLSTM core model.')
    parser.add_argument('--model_core_bilstm_layer_size', type=int, default=256, help='Numer of output units for LSTM layers in biLSTM core model.')
    parser.add_argument('--model_core_bilstm_layer_dropout', type=int, default=2, help='Dropout rate applied to LSTM layers in biLSTM core model.')
    parser.add_argument('--model_core_bilstm_dropout', type=int, default=0.25, help='GaussianDropout rate applied between biLSTM layers in biLSTM core model.')
    parser.add_argument('--model_core_bilstm_noise', type=float, default=0.2, help='GaussianNoise rate applied between biLSTM layers in biLSTM core model.')
    parser.add_argument('--model_core_transformer_input_dropout', type=int, default=0.2, help='Dropout rate applied to input of transformer core model')
    parser.add_argument('--model_core_transformer_hidden_size', type=int, default=32, help='Sublayer hidden size in transformer core model.')
    parser.add_argument('--model_core_transformer_sent_max_length', type=int, default=75, help='Assumed maximum lenght of sentence used to generate positional signal in transformer core model.')
    parser.add_argument('--model_core_transformer_layers', type=int, default=3, help='Number of encoder layers in core transformer model.')
    parser.add_argument('--model_core_transformer_attention_key_dense_size', type=int, default=20, help='Size of attention key sublayers\' dense layer in core transformer model.')
    parser.add_argument('--model_core_transformer_attention_query_dense_size', type=int, default=20, help='Size of attention query sublayers\' dense layer in core transformer model.')
    parser.add_argument('--model_core_transformer_attention_heads', type=int, default=10, help='Number of heads of multi-head attention layer in core transformer model.')
    parser.add_argument('--model_core_transformer_attention_dropout', type=float, default=0.2, help='Dropout rate applied to each attention sublayer in core transformer model.')
    parser.add_argument('--model_core_transformer_pff_filter_size', type=int, default=3, help='Size of filter for positional feed-forward sublayer in core transformer model.')
    parser.add_argument('--model_core_transformer_pff_dropout', type=int, default=0.2, help='Dropout rate applied to each positional feed-forward sublayer in core transformer model.')
    parser.add_argument('--model_core_transformer_layer_dropout', type=int, default=0.2, help='Dropout rate applied to each encoder layer in core transformer model.')

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

class Step:
    def __init__(self, model, losses, weights):
        self._model = model
        self._losses = losses
        self._weights = weights

    def on(self, batch):
        summaries = dict()

        y_pred = self._model(batch.x)
        y_true = batch.y
        
        loss_total = 0.0
        for (feat, yt, yp, l, weight) in zip(y_true.keys(), y_true.values(), y_pred.values(), self._losses.values(), self._weights):
            loss = l(yt, yp)
            loss_total += weight * loss

            summaries['{}_LOSS'.format(feat)] = loss.numpy()

        summaries['TOTAL_LOSS'] = loss_total.numpy()

        return loss_total, summaries, y_pred

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

    log('Creating encoder...')
    encoder = FeaturesEncoder(vocabs, args)

    log('Creating train generator...')
    generator_train = LenwiseBatchGenerator(sents_train, args.batch_size) \
                      if args.batch_lenwise else \
                      RandomBatchGenerator(sents_train, args.batch_size)
    generator_train = map(encoder.encode_batch, generator_train)

    if validation:
        log('Creating dev generator...')
        generator_dev = RandomBatchGenerator(sents_dev, args.batch_size_dev) \
                        if args.batch_size_dev is not None else \
                        AllAtOnceBatchGenerator(sents_dev)
        generator_dev = map(encoder.encode_batch, generator_dev)

    log('Creating model & optimizer...')
    model = ParserModel(args, word_embeddings=embeddings, vocabs=conllu_train.vocabs)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.9, epsilon=1e-4)

    log('Loading checkpoints...')
    checkpoint_path = args.save_dir + '/checkpoints/'
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, optimizer_step=tf.train.get_or_create_global_step())
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    log('Initializing logger...')
    logger = tf.contrib.summary.create_file_writer(args.save_dir + '/logs/')
    logger.set_as_default()

    epoch_start = tf.train.get_or_create_global_step() / args.batch_per_epoch
    epochs_early_stopping_counter = 0

    log('Starting training ({} -> {} epochs, {} batches/epoch)...'.format(epoch_start, args.epochs, args.batch_per_epoch))
    step = Step(model, parser.losses.y(args), args.loss_weights)
    for epoch_i in range(epoch_start, args.epochs):

        log('Epoch {}/{}.'.format(epoch_i + 1, args.epochs))

        for (batch_i, batch) in ((i, next(generator_train)) for i in range(args.batch_per_epoch)):
            global_step = tf.train.get_or_create_global_step()

            with tf.GradientTape() as tape:
                loss_total, summaries, _ = step.on(batch)

            grads = tape.gradient(loss_total, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=global_step)

            # log to console
            if batch_i % args.batch_per_summary == 0:
                log(summaries)

            # log to file
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.batch_per_summary):
                for code, value in summaries.items():
                    tf.contrib.summary.scalar('{}_TRAIN'.format(code), value)

            # initialize with first step - new or checkpointed model
            if 'loss_total_min' not in locals():       
                loss_total_min = loss_total

        if validation:
            log('Validation {}/{}.'.format(epoch_i + 1, args.epochs))
            batch_dev = next(generator_dev)
            loss_total, summaries, y = step.on(batch_dev)

            file_gold = args.save_dir + '/validation/{}_gold.conllu'.format(epoch_i)
            sents_dev_gold = encoder.decode_batch(batch_dev)
            conll.write_conllu(file_gold, sents_dev_gold)

            file_system = args.save_dir + '/validation/{}_system.conllu'.format(epoch_i)
            sents_dev_system = encoder.decode_batch(batch_dev, y)
            conll.write_conllu(file_system, sents_dev_system)

            for code, score in parser.scores.y.items():
                score = score(sents_dev_gold, sents_dev_system)
                summaries[code] = score

            with tf.contrib.summary.always_record_summaries():
                for code, value in summaries.items():
                    tf.contrib.summary.scalar('{}_DEV'.format(code), value)

        # save checkpoint
        if loss_total < loss_total_min:
            log('Saving checkpoint...')
            loss_total_min = loss_total
            checkpoint.save(checkpoint_path + 'model')

            epochs_early_stopping_counter = 0
        else:
            epochs_early_stopping_counter += 1

            if epochs_early_stopping_counter >= args.epochs_early_stopping:
                log('Total loss did not decrease from {} steps. Stopping.'.format(epochs_early_stopping_counter))
                break

    log('Finished training.')

if __name__ == '__main__':
    main()
