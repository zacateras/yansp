import utils
import numpy as np
import keras

class BasePropEncoder:
    def __init__(self, property_selector, padded=True, onehot=False, softmax=False):
        assert not (onehot and softmax)

        self.property_selector = property_selector
        self.padded = padded
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

    def encode_batch(self, source):
        enc = [
            self._encode(sent) \
            for sent in source
        ]
        if self.padded:
            enc = self._to_padded(enc)
        if self.onehot:
            enc = self._to_onehot(enc)
        return enc

    def decode_batch(self, source):
        if self.softmax:
            source = self._from_softmax(source)
        elif self.onehot:
            source = self._from_onehot(source)
        dec = [
            self._decode(enc) \
            for enc in source
        ]
        return dec
    
    def _encode(self, sent):
        raise NotImplementedError

    def _decode(self, enc):
        raise NotImplementedError

    def _to_padded(self, source):
        return keras.preprocessing.sequence.pad_sequences(
            source, dtype=self.dtype, padding='post', truncating='post', value=utils.vocab.PAD_ID)

    def _to_onehot(self, source):
        raise NotImplementedError

    def _from_onehot(self, source):
        return np.argmax(source, axis=-1)

    def _from_softmax(self, source):
        return np.around(source).astype(self.dtype)

class BasePropVocabEncoder(BasePropEncoder):
    def __init__(self, property_selector, vocab: utils.Vocab, padded=True, onehot=False, softmax=False):
        super(BasePropVocabEncoder, self).__init__(property_selector, padded, onehot, softmax)

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

class PropVocabEncoder(BasePropVocabEncoder):
    def __init__(self, property_selector, vocab: utils.Vocab, padded=True, onehot=False, softmax=False):
        super(PropVocabEncoder, self).__init__(property_selector, vocab, padded, onehot, softmax)
    
    def _encode(self, sent):
        return [
            self.vocab.item2id(self.property_selector(word)) \
            for word in sent.words
        ]

    def _decode(self, enc):
        return [
            self.vocab.id2item(id) for id in enc
        ]

class PropIterVocabEncoder(BasePropVocabEncoder):
    def __init__(self, property_selector, vocab: utils.Vocab, max_length: int, padded=True, onehot=False, softmax=False):
        super(PropIterVocabEncoder, self).__init__(property_selector, vocab, padded, onehot, softmax)

        self.max_length = max_length

    def _encode(self, sent):
        return [
            self._encode_word(word) \
            for word in sent.words
        ]

    def _decode(self, enc):
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

class HeadPropEncoder(BasePropEncoder):
    def __init__(self, padded=True, onehot=False, softmax=False):
        super(HeadPropEncoder, self).__init__(lambda x: x.head, padded, onehot=True, softmax=False)

        self.dtype = np.uint8

    def _encode(self, sent):
        return [
            int(self.property_selector(word)) \
            for word in sent.words
        ]

    def _decode(self, enc):
        # TODO mst
        return [
            head \
            for head in enc
        ]

    def _to_onehot(self, source):
        if hasattr(source, '__iter__') and hasattr(source[0], 'shape'):
            num_classes = source[0].shape[0]
        elif hasattr(source, 'shape'):
            num_classes = source.shape[0]
        else:
            num_classes = len(source)

        return keras.utils.to_categorical(source, num_classes, dtype=self.dtype)

class FeatsPropEncoder(BasePropEncoder):
    def __init__(self, vocab: utils.Vocab, padded=True):
        super(FeatsPropEncoder, self).__init__(lambda x: x.feats, padded, onehot=False, softmax=False)

        self.vocab = vocab
        self.dtype = np.uint8
        self.threashold = 0.5

    def _encode(self, sent):
        return [
            self._encode_word(word) \
            for word in sent.words
        ]

    def _decode(self, enc):
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
