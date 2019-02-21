import argparse

import conll
import utils
import models

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
    parser.add_argument('--model_feats_dense_size', type=int, default=100, help='Size of feats model hidden dense size.')
    parser.add_argument('--model_lemma_dense_size', type=int, default=100, help='Size of lemma model hidden dense size.')
    parser.add_argument('--model_dropout', type=float, default=0.25, help='Dropout rate applied on default to dropout layers.')
    parser.add_argument('--model_noise', type=float, default=0.2, help='Noise StdDev applied on default to noise layers.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    conllu_train = conll.load_conllu(args.train_file)
    embeddings = utils.Embeddings.from_file(args.wordvec_file)

    char_vocab = conllu_train.vocabs[conll.vocab.CHAR]
    deprel_vocab = conllu_train.vocabs[conll.vocab.DEPREL]
    upos_vocab = conllu_train.vocabs[conll.vocab.UPOS]
    feats_vocab = conllu_train.vocabs[conll.vocab.FEATS]
    lemma_vocab = conllu_train.vocabs[conll.vocab.LEMMA]
    word_vocab = embeddings.vocab

    for sent in conllu_train.sents:
        enc_word = [
            word_vocab.item2id(word.form) \
            for word in sent.words
        ]

        enc_form_char = [
            [
                char_vocab.item2id(char) \
                for char in word.form
            ] \
            for word in sent.words
        ]

        enc_lemma = [
            lemma_vocab.item2id(word.lemma) \
            for word in sent.words
        ]

        enc_upos = [
            upos_vocab.item2id(word.upos) \
            for word in sent.words
        ]

        pass

    parser = models.ParserModel(
        args,
        word_embeddings=embeddings,
        vocabs=conllu_train.vocabs
    )

    parser.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001, clipvalue=5.0, beta_1=0.9, beta_2=0.9, decay=1e-4)
    )

    pass

if __name__ == '__main__':
    main()
