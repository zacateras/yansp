from utils.encoders import BaseSentEncoder, SentVocabEncoder, SentIterVocabEncoder
from utils.mst import mst

import conll
import conll.vocab

import parser.features as F

import keras
import numpy as np

class Batch:
    def __init__(self, items, x, y):
        self.items = items
        self.x = x
        self.y = y

class HeadSentEncoder(BaseSentEncoder):
    def __init__(self, padded=True, onehot=False, softmax=False):
        super(HeadSentEncoder, self).__init__(lambda x: x.head, padded, onehot=True, softmax=False)

        self.dtype = np.uint8

    def _encode(self, sent):
        return [
            int(self.property_selector(word)) \
            for word in sent.words
        ]

    def _to_onehot(self, source):
        if hasattr(source, '__iter__') and hasattr(source[0], 'shape'):
            num_classes = source[0].shape[0]
        elif hasattr(source, 'shape'):
            num_classes = source.shape[0]
        else:
            num_classes = len(source)

        return keras.utils.to_categorical(source, num_classes, dtype=self.dtype)

    def _decode(self, sent, enc):
        num_words = len(sent)

        # wipe out paddings
        enc = enc[:num_words, :num_words]
        # mst is implemented in a way that this encoding must be transposed
        enc = np.transpose(enc)

        x = mst(enc)

        return x

    def _from_onehot(self, source):
        return source

class FeatsSentEncoder(BaseSentEncoder):
    def __init__(self, vocab, padded=True):
        super(FeatsSentEncoder, self).__init__(lambda x: x.feats, padded, onehot=False, softmax=False)

        self.vocab = vocab
        self.dtype = np.uint8
        self.threashold = 0.5

    def _encode(self, sent):
        return [
            self._encode_word(word) \
            for word in sent.words
        ]

    def _decode(self, sent, enc):
        feats = [
            [
                self.vocab.id2item(_id[0])
                for _id in np.argwhere(word > self.threashold) \
            ] \
            for word in enc
        ]
        
        return feats

    def _encode_word(self, word):
        p = self.property_selector(word)
        enc = np.zeros((self.vocab.size), dtype=self.dtype)

        for feat in p:
            enc[self.vocab.item2id(feat)] = 1

        return enc

class FeaturesEncoder:
    def __init__(self, vocabs, args, x_feats = F.X, y_feats = F.Y):
        _ = dict()

        if conll.vocab.WORD in vocabs:
            _[F.FORM] = SentVocabEncoder(lambda w: w.form, vocabs[conll.vocab.WORD])
        _[F.FORM_CHAR] = SentIterVocabEncoder(lambda w: w.form, vocabs[conll.vocab.CHAR], args.model_word_max_length)
        _[F.LEMMA_CHAR] = SentIterVocabEncoder(lambda w: w.lemma, vocabs[conll.vocab.CHAR], args.model_word_max_length, onehot=True)
        _[F.UPOS] = SentVocabEncoder(lambda w: w.upos, vocabs[conll.vocab.UPOS], onehot=True)
        _[F.DEPREL] = SentVocabEncoder(lambda w: w.deprel, vocabs[conll.vocab.DEPREL], onehot=True)
        _[F.FEATS] = FeatsSentEncoder(vocabs[conll.vocab.FEATS])
        _[F.HEAD] = HeadSentEncoder(onehot=True)

        self.x_encoders = [(f, _[f]) for f in x_feats if f in _]
        self.y_encoders = [(f, _[f]) for f in y_feats if f in _]

    def encode_batch(self, sents):
        return Batch(
            items = sents,
            x = dict((f, encoder.encode_batch(sents)) for (f, encoder) in self.x_encoders),
            y = dict((f, encoder.encode_batch(sents)) for (f, encoder) in self.y_encoders)
        )
    
    def decode_batch(self, batch: Batch, y=None):
        if y is None:
            y = batch.y

        y = dict((f, encoder.decode_batch(batch.items, y[f])) for (f, encoder) in self.y_encoders)

        sents = []

        for sent_i in range(len(batch.items)):
            sent = batch.items[sent_i]

            words = list()
            for word_i in range(len(sent)):
                columns = [
                    str(sent[word_i].id),                 # 1 id
                    sent[word_i].form,                    # 2 form
                    y[F.LEMMA_CHAR][sent_i][word_i],      # 3 lemma
                    y[F.UPOS][sent_i][word_i],            # 4 upos
                    '_',                                  # 5 xpos
                    '|'.join(y[F.FEATS][sent_i][word_i]), # 6 feats
                    str(y[F.HEAD][sent_i][word_i]),       # 7 head
                    y[F.DEPREL][sent_i][word_i],          # 8 deprel
                    '_',                                  # 9 deps
                    '_'                                   # 10 misc
                ]

                words.append(conll.UDWord(columns))

            sents.append(conll.UDSentence(words))

        return sents
