import argparse
import time
import os.path
import re

import tensorflow as tf
import numpy as np

import conll
import utils

from utils.generators import LenwiseBatchGenerator, RandomBatchGenerator, AllAtOnceBatchGenerator
from parser.summary import summary_for_parser
from parser.encoders import FeaturesEncoder
from parser.models import ParserModel
import parser.losses
import parser.scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordvec_file', type=str, help='Input file for word vectors.')
    parser.add_argument('--train_file', type=str, required=True, help='Input CoNLL-U train file.')
    parser.add_argument('--dev_file', type=str, help='Input CoNLL-U dev file.')
    parser.add_argument('--test_file', type=str, help='Input CoNLL-U test file.')
    parser.add_argument('--save_dir', type=str, required=True, help='Root dir for saving logs and models.')

    parser.add_argument('--mode', default='train', type=str, choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')

    parser.add_argument('--epochs', type=int, default=20, help='Numer of epochs.')
    parser.add_argument('--epochs_early_stopping', type=int, default=5, help='Number of epochs w/o loss decrease required for early stopping.')
    parser.add_argument('--checkpoint_rolling', type=bool, default=False, help='If flag is set to true, old checkpoints are deleted every time new one is created.')
    parser.add_argument('--checkpoint_prefix', type=str, default='model', help='Prefix of checkpoint file names.')
    parser.add_argument('--batch_per_epoch', type=int, default=50, help='Number of batches per epoch.')
    parser.add_argument('--batch_per_console_summary', type=int, default=1, help='Summary console logs reporting interval.')
    parser.add_argument('--batch_per_file_summary', type=int, default=1, help='Summary file logs reporting interval.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Size of batches (in words).')
    parser.add_argument('--batch_lenwise', type=bool, default=False, help='If true, sentences will be sorted and processed in length order')
    parser.add_argument('--batch_dev_limit', type=int, default=None, help='Maximum size of batch (in words) during validation phase. If None then whole file is used, otherwise at most `batch_dev_limit` leading tokens.')

    parser.add_argument('--loss_cycle_weight', type=float, default=1.0, help='Relative weight of cycle loss.')
    parser.add_argument('--loss_cycle_n', type=int, default=3, help='Number of cycles to find.')
    parser.add_argument('--loss_weights', default=[0.2, 0.8, 0.05, 0.05, 0.2], help='Losses weights.')

    parser.add_argument('--optimizer_lr', type=float, default=0.001, help='Optimizer learning rate.')
    parser.add_argument('--optimizer_b1', type=float, default=0.9, help='Optimizer first-moment exponential decay rate (beta1).')
    parser.add_argument('--optimizer_b2', type=float, default=0.999, help='Optimizer second-moment exponential decay rate (beta2).')
    parser.add_argument('--optimizer_eps', type=float, default=1e-8, help='Optimizer epsilon.')

    parser.add_argument('--signature_prefix', type=str, default=None, help='Custom model signature prefix.')
    parser.add_argument('--signature_suffix', type=str, default=None, help='Custom model signature suffix.')

    parser.add_argument('--model_word_dense_size', type=int, default=100, help='Size of word model output dense layer.')
    parser.add_argument('--model_word_max_length', type=int, default=30, help='Maximum length of words.')
    parser.add_argument('--model_char_embedding_dim', type=int, default=100, help='Dimension of character level embeddings.')
    parser.add_argument('--model_char_conv_layers', type=int, default=3, help='Number of convolution layers in character model.')
    parser.add_argument('--model_char_conv_size', type=int, default=30, help='Size of character model convolution layers.')
    parser.add_argument('--model_char_dense_size', type=int, default=100, help='Size of character model output dense layer.')

    parser.add_argument('--model_core_type', default='transformer', type=str, choices=['transformer', 'biLSTM'], help='Type of core model used (either transformer or biLSTM).')
    parser.add_argument('--model_core_bilstm_layers', type=int, default=2, help='Numer of LSTM layers in biLSTM core model.')
    parser.add_argument('--model_core_bilstm_layer_size', type=int, default=256, help='Numer of output units for LSTM layers in biLSTM core model.')
    parser.add_argument('--model_core_bilstm_layer_dropout', type=float, default=0.2, help='Dropout rate applied to LSTM layers in biLSTM core model.')
    parser.add_argument('--model_core_bilstm_dropout', type=float, default=0.25, help='GaussianDropout rate applied between biLSTM layers in biLSTM core model.')
    parser.add_argument('--model_core_bilstm_noise', type=float, default=0.2, help='GaussianNoise rate applied between biLSTM layers in biLSTM core model.')
    parser.add_argument('--model_core_transformer_input_dropout', type=float, default=0.2, help='Dropout rate applied to input of transformer core model')
    parser.add_argument('--model_core_transformer_hidden_size', type=int, default=32, help='Sublayer hidden size in transformer core model.')
    parser.add_argument('--model_core_transformer_sent_max_length', type=int, default=75, help='Assumed maximum lenght of sentence used to generate positional signal in transformer core model.')
    parser.add_argument('--model_core_transformer_layers', type=int, default=3, help='Number of encoder layers in core transformer model.')
    parser.add_argument('--model_core_transformer_attention_heads', type=int, default=10, help='Number of heads of multi-head attention layer in core transformer model.')
    parser.add_argument('--model_core_transformer_attention_key_dense_size', type=int, default=20, help='Size of attention key sublayers\' dense layer in core transformer model.')
    parser.add_argument('--model_core_transformer_attention_query_dense_size', type=int, default=20, help='Size of attention query sublayers\' dense layer in core transformer model.')
    parser.add_argument('--model_core_transformer_attention_dropout', type=float, default=0.2, help='Dropout rate applied to each attention sublayer in core transformer model.')
    parser.add_argument('--model_core_transformer_pff_filter_size', type=int, default=3, help='Size of filter for positional feed-forward sublayer in core transformer model.')
    parser.add_argument('--model_core_transformer_pff_dropout', type=float, default=0.2, help='Dropout rate applied to each positional feed-forward sublayer in core transformer model.')
    parser.add_argument('--model_core_transformer_layer_dropout', type=float, default=0.2, help='Dropout rate applied to each encoder layer in core transformer model.')

    parser.add_argument('--model_head_dense_size', type=int, default=100, help='Size of head model hidden dense size.')
    parser.add_argument('--model_deprel_dense_size', type=int, default=100, help='Size of deprel model hidden dense size.')
    parser.add_argument('--model_upos_dense_size', type=int, default=100, help='Size of UPOS model hidden dense size.')
    parser.add_argument('--model_feats_max_length', type=int, default=100, help='Maximum length of features.')
    parser.add_argument('--model_feats_dense_size', type=int, default=100, help='Size of feats model hidden dense size.')
    parser.add_argument('--model_lemma_dense_size', type=int, default=25, help='Size of lemma model hidden dense size.')

    parser.add_argument('--model_dropout', type=float, default=0.25, help='Dropout rate applied on default to dropout layers.')
    parser.add_argument('--model_noise', type=float, default=0.2, help='Noise StdDev applied on default to noise layers.')

    args = parser.parse_args()
    return args

def model_signature_from_args(args):
    args_vars = vars(args)

    # model parametrization properties
    p = (x for x in zip(args_vars.keys(), args_vars.values()) if x[0].startswith('model_'))

    # model meaningful parametrization properties
    pm = [args.model_core_type]

    if args.model_core_type == 'transformer':
        pm.append(str(args.model_core_transformer_layers))
        pm.append(str(args.model_core_transformer_hidden_size))
        pm.append('ah' + str(args.model_core_transformer_attention_heads))
        pm.append('ak' + str(args.model_core_transformer_attention_key_dense_size))
        pm.append('aq' + str(args.model_core_transformer_attention_query_dense_size))
        p = (x for x in p if not (x[0].startswith('model_core') and not x[0].startswith('model_core_transformer')))

    elif args.model_core_type == 'biLSTM':
        pm.append(str(args.model_core_bilstm_layers))
        pm.append(str(args.model_core_bilstm_layer_size))
        p = (x for x in p if not (x[0].startswith('model_core') and not x[0].startswith('model_core_bilstm')))

    p = list(p)

    # model paramerization hash string
    h = utils.genkey(str(sorted(p, key=lambda x: x[0])))

    if args.signature_prefix is not None:
        pm.insert(0, args.signature_prefix)
    if args.signature_suffix is not None:
        pm.append(args.signature_suffix)

    return '.'.join(pm) + '-' + h, p

def model_conf_save(file, model_conf, model_variables):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w+') as f:
        f.write('# configuration\n')
        for x in model_conf:
            f.write('{}={}\n'.format(x[0], x[1]))
        
        f.write('\n# variables\n')
        for k, v in model_variables.items():
            f.write('{}={}\n'.format(k, v))

def log(message):
    print('{} {}'.format(time.strftime("%Y-%m-%d %H:%M:%S"), message))

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

def main():
    tf.enable_eager_execution()

    log('Parsing args...')
    args = parse_args()
    validation = True if args.dev_file is not None else False
    signature, model_conf = model_signature_from_args(args)
    base_dir = os.path.join(args.save_dir, signature)
    
    log('Loading CoNLL-U training file...')
    conllu_train = conll.load_conllu(args.train_file)
    sents_train = [sent.with_root() for sent in conllu_train.sents]

    if validation:
        log('Loading CoNLL-U validation file...')
        conllu_dev = conll.load_conllu(args.dev_file)
        sents_dev = [sent.with_root() for sent in conllu_dev.sents]

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
        generator_dev = AllAtOnceBatchGenerator(sents_dev, args.batch_dev_limit)
        generator_dev = map(encoder.encode_batch, generator_dev)

    log('Creating model & optimizer...')
    model = ParserModel(args, word_embeddings=embeddings, vocabs=conllu_train.vocabs)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=args.optimizer_lr,
        beta1=args.optimizer_b1,
        beta2=args.optimizer_b2,
        epsilon=args.optimizer_eps)

    log('Loading checkpoints...')
    checkpoint_prefix = args.checkpoint_prefix
    checkpoint_path = base_dir + '/checkpoints/'
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, optimizer_step=tf.train.get_or_create_global_step())
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    log('Initializing logger...')
    logger = tf.contrib.summary.create_file_writer(base_dir + '/logs/')
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
                loss_total, losses, _ = step.on(batch)

            grads = tape.gradient(loss_total, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=global_step)

            # log to console
            if batch_i % args.batch_per_console_summary == 0:
                log(losses)

            # log to file
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.batch_per_file_summary):
                for code, value in losses.items():
                    tf.contrib.summary.scalar('train_loss/{}'.format(code), value)

            # initialize with first step - new or checkpointed model
            if 'loss_total_min' not in locals():    
                loss_total_min = loss_total

                model_conf_file = os.path.join(base_dir, 'model.conf')
                if not os.path.exists(model_conf_file):
                    log('Saving model parameters summary...')
                    model_conf_save(model_conf_file, model_conf, summary_for_parser(model))

        if validation:
            log('Validation {}/{}.'.format(epoch_i + 1, args.epochs))
            batch_dev = next(generator_dev)
            loss_total, losses, y = step.on(batch_dev)

            file_gold = base_dir + '/validation/{}_gold.conllu'.format(epoch_i)
            sents_dev_gold = encoder.decode_batch(batch_dev)
            conll.write_conllu(file_gold, sents_dev_gold)

            file_system = base_dir + '/validation/{}_system.conllu'.format(epoch_i)
            sents_dev_system = encoder.decode_batch(batch_dev, y)
            conll.write_conllu(file_system, sents_dev_system)

            summaries = dict()

            for code, loss in losses.items():
                summaries['dev_loss/{}'.format(code)] = loss

            for code, score in parser.scores.y.items():
                score = score(sents_dev_gold, sents_dev_system)
                summaries['dev_score/{}'.format(code)] = score

            for code, score in conll.evaluate(file_gold, file_system).items():
                summaries['dev_conll/{}/f1'.format(code)] = score.f1

            with tf.contrib.summary.always_record_summaries():
                for code, value in summaries.items():
                    tf.contrib.summary.scalar(code, value)

        # save checkpoint
        if loss_total < loss_total_min:
            log('Saving checkpoint...')
            loss_total_min = loss_total
            checkpoint.save(checkpoint_path + checkpoint_prefix)

            if args.checkpoint_rolling:
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

            if epochs_early_stopping_counter >= args.epochs_early_stopping:
                log('Total loss did not decrease from {} steps. Stopping.'.format(epochs_early_stopping_counter))
                break

    log('Finished training.')

if __name__ == '__main__':
    main()
