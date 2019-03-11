import utils
import numpy as np
import keras

class BaseSentEncoder:
    def __init__(self, property_selector, padded=True, onehot=False, softmax=False, pad_with=utils.vocab.PAD_ID):
        assert not (onehot and softmax)

        self.property_selector = property_selector
        self.padded = padded
        self.pad_with = pad_with
        self.onehot = onehot
        self.softmax = softmax
        self.dtype = np.float32

    def encode(self, sent):
        enc = self._encode(sent)
        if self.padded:
            enc = self._to_padded(enc)
        if self.onehot:
            enc = self._to_onehot(enc)
        return enc

    def encode_batch(self, sents):
        encs = [
            self._encode(item) \
            for item in sents
        ]
        if self.padded:
            encs = self._to_padded(encs)
        if self.onehot:
            encs = self._to_onehot(encs)
        return encs
    
    def _encode(self, sent):
        raise NotImplementedError

    def _to_padded(self, source):
        return keras.preprocessing.sequence.pad_sequences(
            source, dtype=self.dtype, padding='post', truncating='post', value=self.pad_with)

    def _to_onehot(self, source):
        raise NotImplementedError


    def decode_batch(self, sents, encs):
        if self.softmax:
            encs = self._from_softmax(encs)
        elif self.onehot:
            encs = self._from_onehot(encs)
        decs = [
            self._decode(sent, enc) \
            for sent, enc in zip(sents, encs)
        ]
        return decs

    def _decode(self, sent, enc):
        raise NotImplementedError

    def _from_onehot(self, source):
        return np.argmax(source, axis=-1)

    def _from_softmax(self, source):
        return np.around(source).astype(self.dtype)

class BaseSentVocabEncoder(BaseSentEncoder):
    def __init__(self, property_selector, vocab: utils.Vocab, padded=True, onehot=False, softmax=False):
        super(BaseSentVocabEncoder, self).__init__(property_selector, padded, onehot, softmax)

        self.vocab = vocab

        # numpy type ranges:
        # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
        if vocab.size < 256:
            self.dtype = np.uint8
        elif vocab.size < 65536:
            self.dtype = np.uint16
        elif vocab.size < 4294967296:
            self.dtype = np.uint32
        else:
            self.dtype = np.uint64

    def _to_onehot(self, source):
        return keras.utils.to_categorical(
            y=source, num_classes=self.vocab.size, dtype=self.dtype)

class SentVocabEncoder(BaseSentVocabEncoder):
    def __init__(self, property_selector, vocab: utils.Vocab, padded=True, onehot=False, softmax=False):
        super(SentVocabEncoder, self).__init__(property_selector, vocab, padded, onehot, softmax)
    
    def _encode(self, sent):
        return [
            self.vocab.item2id(self.property_selector(word)) \
            for word in sent.words
        ]

    def _decode(self, sent, enc):
        return [
            self.vocab.id2item(id) for id in enc
        ]

class SentIterVocabEncoder(BaseSentVocabEncoder):
    def __init__(self, property_selector, vocab: utils.Vocab, max_length: int, padded=True, onehot=False, softmax=False):
        super(SentIterVocabEncoder, self).__init__(property_selector, vocab, padded, onehot, softmax)

        self.max_length = max_length

    def _encode(self, sent):
        return [
            self._encode_word(word) \
            for word in sent.words
        ]

    def _decode(self, sent, enc):
        return [
            self._decode_word(word) \
            for word in enc
        ]

    def _encode_word(self, word):
        p = self.property_selector(word)
        p_len = len(p)
        enc = []

        for i in range(self.max_length):
            if i < p_len:
                enc.append(self.vocab.item2id(p[i]))
            else:
                enc.append(utils.vocab.PAD_ID)

        return enc

    def _decode_word(self, word):
        word = (self.vocab.id2item(id) for id in word if id != utils.vocab.PAD_ID)
        word = ''.join(word)
        
        return word
