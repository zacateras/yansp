import argparse

import conll
import utils
import models

import tf

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
    parser.add_argument('--model_lstm_layers', type=int, default=2, help='Numer of LSTM layers in core model.')
    parser.add_argument('--model_lstm_units', type=int, default=512, help='Numer of output units for LSTM layers in core model.')
    parser.add_argument('--model_lstm_dropout', type=int, default=2, help='Dropout rate applied to LSTM layers in core model.')
    parser.add_argument('--model_dropout', type=float, default=0.25, help='Dropout rate applied on default to dropout layers.')
    parser.add_argument('--model_noise', type=float, default=0.2, help='Noise StdDev applied on default to noise layers.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    conllu_train = conll.load_conllu(args.train_file)

    embeddings = utils.Embeddings.from_file(args.wordvec_file)

    word_model = models.WordModel(
        embeddings=embeddings,
        dense_size=args.model_word_dense_size,
        dropout=args.model_dropout
    )

    core_model = models.CoreModel(
        lstm_layers=args.model_lstm_layers,
        lstm_units=args.model_lstm_units,
        lstm_dropout=args.model_lstm_dropout,
        dropout=args.model_dropout,
        noise=args.model_noise
    )

    pass

if __name__ == '__main__':
    main()
