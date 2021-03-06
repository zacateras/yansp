import argparse
import os.path
import re
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd

import conll
import utils
from utils import log

from utils.generators import LenwiseBatchGenerator, RandomBatchGenerator, OneshotBatchGenerator
from parser.encoders import FeaturesEncoder
from parser.models import ParserModel
import parser.losses
import parser.scores
import conf

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    def add_train_arguments(parser):
        parser.set_defaults(mode='train')
        parser.add_argument('--save_dir', type=str, required=True, help='Root dir for saving logs and models.')
        parser.add_argument('--train_file', type=str, required=True, help='Input CoNLL-U train file.')
        parser.add_argument('--dev_file', type=str, help='Input CoNLL-U dev file.')
        parser.add_argument('--test_file', type=str, help='Input CoNLL-U test file.')
        parser.add_argument('--wordvec_file', type=str, help='Input file for word vectors.')

        parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
        parser.add_argument('--epochs_early_stopping', type=int, default=5, help='Number of epochs w/o loss decrease required for early stopping.')
        parser.add_argument('--checkpoint_rolling', type=utils.str2bool, default=False, help='If flag is set to true, old checkpoints are deleted every time new one is created.')
        parser.add_argument('--checkpoint_prefix', type=str, default='model', help='Prefix of checkpoint file names.')
        parser.add_argument('--batch_per_epoch', type=int, default=50, help='Number of batches per epoch.')
        parser.add_argument('--batch_per_console_summary', type=int, default=1, help='Summary console logs reporting interval.')
        parser.add_argument('--batch_per_file_summary', type=int, default=1, help='Summary file logs reporting interval.')
        parser.add_argument('--batch_size', type=int, default=1000, help='Size of batches (in words).')
        parser.add_argument('--batch_lenwise', type=utils.str2bool, default=False, help='If true, sentences will be sorted and processed in length order')
        parser.add_argument('--batch_size_dev', type=int, default=1000, help='Size of batches (in words) during validation phase. If None then whole file is used.')
        parser.add_argument('--batch_limit_dev', type=int, default=None, help='Maximum size (in words) of validation data. If None then whole file is used.')

        parser.add_argument('--loss_cycle_weight', type=float, default=1.0, help='Relative weight of cycle loss.')
        parser.add_argument('--loss_cycle_n', type=int, default=3, help='Number of cycles to find.')
        parser.add_argument('--loss_weights', default=[0.25, 0.5, 0.1, 0.2, 0.25], help='Losses weights.')

        parser.add_argument('--signature_prefix', type=str, default=None, help='Custom model signature prefix.')
        parser.add_argument('--signature_suffix', type=str, default=None, help='Custom model signature suffix.')

        parser.add_argument('--optimizer_lr', type=float, default=0.001, help='Optimizer learning rate.')
        parser.add_argument('--optimizer_b1', type=float, default=0.9, help='Optimizer first-moment exponential decay rate (beta1).')
        parser.add_argument('--optimizer_b2', type=float, default=0.999, help='Optimizer second-moment exponential decay rate (beta2).')
        parser.add_argument('--optimizer_eps', type=float, default=1e-8, help='Optimizer epsilon.')

        parser.add_argument('--model_inputs', nargs='+', default=['word', 'char'], help='Used input (also features) types.')
        parser.add_argument('--model_word_dense_size', type=utils.str2int, default=100, help='Size of word model output dense layer. If `none` then no layer is used.')
        parser.add_argument('--model_word_max_length', type=int, default=30, help='Maximum length of words.')
        parser.add_argument('--model_char_embedding_dim', type=int, default=100, help='Dimension of character level embeddings.')
        parser.add_argument('--model_char_conv_layers', type=int, default=3, help='Number of convolution layers in character model.')
        parser.add_argument('--model_char_conv_size', type=int, default=30, help='Size of character model convolution layers.')

        parser.add_argument('--model_core_type', default='transformer', type=str, choices=['transformer', 'biLSTM'], help='Type of core model used (either transformer or biLSTM).')
        parser.add_argument('--model_core_bilstm_layers', type=int, default=2, help='Numer of LSTM layers in biLSTM core model.')
        parser.add_argument('--model_core_bilstm_layer_size', type=int, default=256, help='Numer of output units for LSTM layers in biLSTM core model.')
        parser.add_argument('--model_core_bilstm_layer_dropout', type=float, default=0.2, help='Dropout rate applied to LSTM layers in biLSTM core model.')
        parser.add_argument('--model_core_bilstm_dropout', type=float, default=0.25, help='GaussianDropout rate applied between biLSTM layers in biLSTM core model.')
        parser.add_argument('--model_core_bilstm_noise', type=float, default=0.2, help='GaussianNoise rate applied between biLSTM layers in biLSTM core model.')
        parser.add_argument('--model_core_transformer_input_dropout', type=float, default=0.2, help='Dropout rate applied to input of transformer core model.')
        parser.add_argument('--model_core_transformer_use_embedding_projection', type=utils.str2bool, default=True, help='Flag enabling internal embedding projection layer.')
        parser.add_argument('--model_core_transformer_use_timing_signal', type=utils.str2bool, default=True, help='Flag enabling timing signal component.')
        parser.add_argument('--model_core_transformer_hidden_size', type=int, default=32, help='Sublayer hidden size in transformer core model.')
        parser.add_argument('--model_core_transformer_sent_max_length', type=int, default=75, help='Assumed maximum lenght of sentence used to generate positional signal in transformer core model.')
        parser.add_argument('--model_core_transformer_layers_direction', type=str, default=['H'], choices=['H', 'V'], help='Direction of stacking encoder layers in core transformer model (horizontal / vertical).')
        parser.add_argument('--model_core_transformer_layers', type=int, default=3, help='Number of encoder layers in core transformer model.')
        parser.add_argument('--model_core_transformer_attention_heads', type=int, default=10, help='Number of heads of multi-head attention layer in core transformer model.')
        parser.add_argument('--model_core_transformer_attention_key_dense_size', type=int, default=20, help='Size of attention key sublayers\' dense layer in core transformer model.')
        parser.add_argument('--model_core_transformer_attention_value_dense_size', type=int, default=20, help='Size of attention query sublayers\' dense layer in core transformer model.')
        parser.add_argument('--model_core_transformer_attention_dropout', type=float, default=0.2, help='Dropout rate applied to each attention sublayer in core transformer model.')
        parser.add_argument('--model_core_transformer_pff_layers', type=int, default=2, help='Number of layers for positional feed-forward sublayer in core transformer model.')
        parser.add_argument('--model_core_transformer_pff_filter_size', type=int, default=32, help='Size of filter for positional feed-forward sublayer in core transformer model.')
        parser.add_argument('--model_core_transformer_pff_dropout', type=float, default=0.2, help='Dropout rate applied to each positional feed-forward sublayer in core transformer model.')
        parser.add_argument('--model_core_transformer_layer_dropout', type=float, default=0.2, help='Dropout rate applied to each encoder layer in core transformer model.')

        parser.add_argument('--model_outputs', nargs='+', default=['lemma', 'upos', 'feats', 'head', 'deprel'], help='Used output (also losses) types.')
        parser.add_argument('--model_head_dense_size', type=int, default=100, help='Size of head model hidden dense size.')
        parser.add_argument('--model_deprel_dense_size', type=int, default=100, help='Size of deprel model hidden dense size.')
        parser.add_argument('--model_upos_dense_size', type=int, default=100, help='Size of UPOS model hidden dense size.')
        parser.add_argument('--model_feats_dense_size', type=int, default=100, help='Size of feats model hidden dense size.')
        parser.add_argument('--model_lemma_dense_size', type=int, default=25, help='Size of lemma model hidden dense size.')

        parser.add_argument('--model_dropout', type=float, default=0.25, help='Dropout rate applied on default to dropout layers.')
        parser.add_argument('--model_noise', type=float, default=0.2, help='Noise StdDev applied on default to noise layers.')

    def add_retrain_arguments(parser):
        # retrain arguments should not have default values - not-None arguments replace train configuration
        parser.set_defaults(mode='retrain')
        parser.add_argument('--model_dir', type=str, required=True, help='Root dir for model configuration, vocabs and checkpoints.')
        parser.add_argument('--train_file', type=str, help='Input CoNLL-U train file.')
        parser.add_argument('--dev_file', type=str, help='Input CoNLL-U dev file.')
        parser.add_argument('--test_file', type=str, help='Input CoNLL-U test file.')

        parser.add_argument('--epochs', type=int, help='Number of epochs.')
        parser.add_argument('--epochs_early_stopping', type=int, help='Number of epochs w/o loss decrease required for early stopping.')
        parser.add_argument('--checkpoint_rolling', type=utils.str2bool, help='If flag is set to true, old checkpoints are deleted every time new one is created.')
        parser.add_argument('--checkpoint_prefix', type=str, help='Prefix of checkpoint file names.')
        parser.add_argument('--batch_per_epoch', type=int, help='Number of batches per epoch.')
        parser.add_argument('--batch_per_console_summary', type=int, help='Summary console logs reporting interval.')
        parser.add_argument('--batch_per_file_summary', type=int, help='Summary file logs reporting interval.')
        parser.add_argument('--batch_size', type=int, help='Size of batches (in words).')
        parser.add_argument('--batch_lenwise', type=utils.str2bool, help='If true, sentences will be sorted and processed in length order')
        parser.add_argument('--batch_size_dev', type=int, help='Size of batches (in words) during validation phase. If None then whole file is used.')
        parser.add_argument('--batch_limit_dev', type=int, help='Maximum size (in words) of validation data. If None then whole file is used.')

        parser.add_argument('--loss_cycle_weight', type=float, help='Relative weight of cycle loss.')
        parser.add_argument('--loss_cycle_n', type=int, help='Number of cycles to find.')
        parser.add_argument('--loss_weights', help='Losses weights.')

    def add_evaluate_arguments(parser):
        parser.set_defaults(mode='evaluate')
        parser.add_argument('--model_dir', type=str, required=True, help='Root dir for model configuration, vocabs and checkpoints.')
        parser.add_argument('--conllu_file', nargs='+', help='Input CoNLL-U file(s).')
        parser.add_argument('--scores_file', type=str, help='Evaluation scores file.')

    add_train_arguments(subparsers.add_parser('train'))
    add_retrain_arguments(subparsers.add_parser('retrain'))
    add_evaluate_arguments(subparsers.add_parser('evaluate'))

    args = parser.parse_args()

    return args

def build(params, vocabs):
    log('Building model...')
    checkpoint_prefix = params['checkpoint_prefix']
    checkpoint_path = params['base_dir'] + '/checkpoints/'
    checkpoint_latest = tf.train.latest_checkpoint(checkpoint_path)

    model = ParserModel(params, vocabs=vocabs, has_checkpoint=(checkpoint_latest is not None))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params['optimizer_lr'],
        beta1=params['optimizer_b1'],
        beta2=params['optimizer_b2'],
        epsilon=params['optimizer_eps'])

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, optimizer_step=tf.train.get_or_create_global_step())

    if checkpoint_latest is not None:
        log('Loading checkpoints...')
        checkpoint.restore(checkpoint_latest)

    log('Creating encoder...')
    encoder = FeaturesEncoder(vocabs, params)

    return model, optimizer, encoder, checkpoint, checkpoint_path, checkpoint_prefix, checkpoint_latest

class Step:
    def __init__(self, model, losses, weights):
        self._model = model
        self._losses = losses
        self._weights = weights

    def on(self, batch):
        losses = dict()

        y_pred = self._model(batch.x)
        y_true = batch.y
        
        loss_total = 0.0
        for (feat, yt, yp, l, weight) in zip(y_true.keys(), y_true.values(), y_pred.values(), self._losses.values(), self._weights):
            loss = l(yt, yp)
            loss_total += weight * loss

            losses[feat] = loss.numpy()

        losses['total'] = loss_total.numpy()

        return loss_total, losses, y_pred

def train(params):
    log('Loading CoNLL-U training file...')
    conllu_train = conll.load_conllu(params['train_file'])
    sents_train = [sent.with_root() for sent in conllu_train.sents]

    validation = True if params['dev_file'] is not None else False
    if validation:
        log('Loading CoNLL-U validation file...')
        conllu_dev = conll.load_conllu(params['dev_file'])
        sents_dev = [sent.with_root() for sent in conllu_dev.sents]
    
    log('Loading vocabs...')
    vocabs = utils.Vocab.load(params['base_dir'], default=conllu_train.vocabs)

    model, optimizer, encoder, checkpoint, checkpoint_path, checkpoint_prefix, checkpoint_latest = build(params, vocabs)

    log('Creating train generator...')
    generator_train = LenwiseBatchGenerator(sents_train, params['batch_size']) \
                      if params['batch_lenwise'] else \
                      RandomBatchGenerator(sents_train, params['batch_size'])
    generator_train = map(encoder.encode_batch, generator_train)

    log('Initializing logger...')
    logger = tf.contrib.summary.create_file_writer(params['base_dir'] + '/logs/')
    logger.set_as_default()

    epoch_start = tf.train.get_or_create_global_step() / params['batch_per_epoch']
    epochs_early_stopping_counter = 0

    log('Starting training ({} -> {} epochs, {} batches/epoch)...'.format(epoch_start, params['epochs'], params['batch_per_epoch']))
    step = Step(model, parser.losses.y(params), params['loss_weights'])

    for epoch_i in range(epoch_start, params['epochs']):

        log('Epoch {}/{}.'.format(epoch_i + 1, params['epochs']))

        for (batch_i, batch) in ((i, next(generator_train)) for i in range(params['batch_per_epoch'])):
            global_step = tf.train.get_or_create_global_step()

            with tf.GradientTape() as tape:
                loss_total, losses, _ = step.on(batch)

            grads = tape.gradient(loss_total, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=global_step)

            # log to console
            if batch_i % params['batch_per_console_summary'] == 0:
                log(losses)

            # log to file
            with tf.contrib.summary.record_summaries_every_n_global_steps(params['batch_per_file_summary']):
                for code, value in losses.items():
                    tf.contrib.summary.scalar('train_loss/{}'.format(code), value)

            # initialize with first step
            if checkpoint_latest is None and 'initialized' not in locals():
                log('Saving model configuration...')
                conf.ensure_saved(params, model)
                utils.Vocab.ensure_saved(params['base_dir'], vocabs)
                checkpoint.save(checkpoint_path + checkpoint_prefix)
                initialized = True

        if validation:
            log('Validation {}/{}.'.format(epoch_i + 1, params['epochs']))
            loss_total_dev, _ = validate(step, encoder, params, sents_dev, '{}/validation/{}_'.format(params['base_dir'], epoch_i))

        # save checkpoint
        if 'loss_total_min_dev' not in locals():
            loss_total_min_dev = loss_total_dev

        if loss_total_dev <= loss_total_min_dev:
            log('Saving checkpoint...')
            loss_total_min_dev = loss_total_dev
            checkpoint.save(checkpoint_path + checkpoint_prefix)

            if params['checkpoint_rolling']:
                lines = []
                with open(os.path.join(checkpoint_path, 'checkpoint')) as f:
                    for line in f:
                        lines.append(line)
                
                checkpoint_prefix_current_match = re.match(r'^model_checkpoint_path\: \".*({}\-[0-9]+)\"'.format(checkpoint_prefix), lines[0])
                checkpoint_prefix_current = checkpoint_prefix_current_match.groups()[0]

                checkpoint_to_delete = (
                    x for x in os.listdir(checkpoint_path) if (
                        not x.startswith('checkpoint') and \
                        not x.startswith(checkpoint_prefix_current)))

                for to_delete in checkpoint_to_delete:
                    to_delete = os.path.join(checkpoint_path, to_delete)
                    log('Removing old checkpoint file {}...'.format(to_delete))
                    os.remove(to_delete)

            epochs_early_stopping_counter = 0
        else:
            epochs_early_stopping_counter += 1

            if epochs_early_stopping_counter >= params['epochs_early_stopping']:
                log('DEV total loss did not decrease from {} steps. Stopping.'.format(epochs_early_stopping_counter))
                break

    log('Finished training.')

def validate(step, encoder, params, sents, out_conllu_prefix):
    generator = OneshotBatchGenerator(sents, params['batch_size_dev'], params['batch_limit_dev'])
    generator = map(encoder.encode_batch, generator)

    loss_total = []
    losses = dict()
    sents_gold = []
    sents_system = []

    for batch in generator:
        loss_total_batch, losses_batch, y = step.on(batch)

        loss_total.append(loss_total_batch)
        for loss_batch_key, loss_batch_value in losses_batch.items():
            if loss_batch_key in losses:
                losses[loss_batch_key].append(loss_batch_value)
            else:
                losses[loss_batch_key] = [loss_batch_value]

        sents_gold = sents_gold + encoder.decode_batch(batch)
        sents_system = sents_system + encoder.decode_batch(batch, y)

    loss_total = np.average(loss_total)
    for loss_key_dev in losses.keys():
        losses[loss_key_dev] = np.average(losses[loss_key_dev])

    file_gold = '{}gold.conllu'.format(out_conllu_prefix)
    conll.write_conllu(file_gold, sents_gold)

    file_system = '{}system.conllu'.format(out_conllu_prefix)
    conll.write_conllu(file_system, sents_system)

    summaries = dict()

    for code, loss in losses.items():
        summaries['dev_loss/{}'.format(code)] = loss

    for code, score in parser.scores.y.items():
        score = score(sents_gold, sents_system)
        summaries['dev_score/{}'.format(code)] = score

    for code, score in conll.evaluate(file_gold, file_system).items():
        summaries['dev_conll/{}/f1'.format(code)] = score.f1

    with tf.contrib.summary.always_record_summaries():
        for code, value in summaries.items():
            tf.contrib.summary.scalar(code, value)

    log(summaries)

    return loss_total, summaries

def evaluate(params):
    # for backward compatibility TODO: remove
    if 'base_dir' not in params:
        params['base_dir'] = params['model_dir']

    for conllu_file_path in params['conllu_file']:
        signature = os.path.split(params['base_dir'])
        signature = signature[len(signature) - 1]

        ud_file = os.path.split(conllu_file_path)
        ud_file = ud_file[len(ud_file) - 1]

        if 'scores_file' in params:
            try:
                scores = pd.read_csv(params['scores_file'])
                scores = scores[scores['signature'] == signature]
                scores = scores[scores['ud_file'] == ud_file]

                if scores['signature'].any():
                    log('Skipping evaluation for {}, {}, because results already exist.'.format(ud_file, signature))
                    continue
            except FileNotFoundError:
                pass

        # build model just once
        if 'vocabs' not in locals():
            log('Loading vocabs...')
            vocabs = utils.Vocab.load(params['base_dir'])

            model, _, encoder, _, _, _, _ = build(params, vocabs)

        log('Loading CoNLL-U file {}...'.format(conllu_file_path))
        conllu_file = conll.load_conllu(conllu_file_path)
        sents = [sent.with_root() for sent in conllu_file.sents]

        log('Evaluating {}...'.format(conllu_file))
        try:
            t_start = datetime.datetime.now()
            step = Step(model, parser.losses.y(params), params['loss_weights'])
            _, summaries = validate(step, encoder, params, sents, '{}/'.format(params['base_dir']))
            t_end = datetime.datetime.now()

            summaries['duration'] = str(t_end - t_start)
            summaries['success'] = True
            log('Evaluated {} successfully.'.format(conllu_file))
        except BaseException as e:
            summaries = dict()
            summaries['success'] = False
            log('Evaluation of {} failed.'.format(conllu_file))

        # directory name treated as signature
        if 'scores_file' in params:
            log('Writing summary for {} to {}...'.format(ud_file, params['scores_file']))

            summaries['signature'] = signature
            summaries['ud_file'] = ud_file
            summaries['ud_version'] = '2.3'
            summaries['timestamp'] = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"

            record = pd.io.json.json_normalize(summaries)

            try:
                scores = pd.read_csv(params['scores_file'])

                scores = scores.append(record)
                scores.to_csv(params['scores_file'], index=False)
            except FileNotFoundError:
                record.to_csv(params['scores_file'], index=False)

def main():
    tf.enable_eager_execution()

    log('Parsing args...')
    params = conf.preprocess(parse_args())

    if params['mode'] in ('train', 'retrain'):
        train(params)

    elif params['mode'] in ('evaluate'):
        evaluate(params)

    else:
        raise RuntimeError('Unexpected mode \'{}\'.'.format(params['mode']))

if __name__ == '__main__':
    main()
