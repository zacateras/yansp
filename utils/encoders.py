import utils
import numpy as np
import tensorflow as tf

class BasePropEncoder:
    def __init__(self, property_selector):
        self.property_selector = property_selector

    def encode(self, source, is_batch):
        return self._encode_batch(source) if is_batch else self._encode_single(source)

    def _encode_batch(self, source):
        raise NotImplementedError
    
    def _encode_single(self, source):
        raise NotImplementedError

    def encode_onehot(self, source, is_batch):
        raise NotImplementedError

class BasePropVocabEncoder(BasePropEncoder):
    def __init__(self, property_selector, vocab: utils.Vocab):
        super(BasePropVocabEncoder, self).__init__(property_selector)

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

    def encode_onehot(self, source, is_batch):
        return tf.keras.utils.to_categorical(
            y=self.encode(source, is_batch),
            num_classes=self.vocab.size,
            dtype=self.dtype
        )

class PropVocabEncoder(BasePropVocabEncoder):
    def __init__(self, property_selector, vocab: utils.Vocab):
        super(PropVocabEncoder, self).__init__(property_selector, vocab)
    
    def _encode_single(self, source):
        return np.fromiter((
            self.vocab.item2id(self.property_selector(word)) \
            for word in source.words
        ), dtype=self.dtype)

    def _encode_batch(self, source):
        return tf.keras.preprocessing.sequence.pad_sequences([
            [
                self.vocab.item2id(self.property_selector(word)) \
                for word in sent.words
            ] \
            for sent in source
        ], dtype=self.dtype, padding='post', truncating='post', value=utils.vocab.PAD_ID)

class PropIterVocabEncoder(BasePropVocabEncoder):
    def __init__(self, property_selector, vocab: utils.Vocab, max_length: int):
        super(PropIterVocabEncoder, self).__init__(property_selector, vocab)

        self.max_length = max_length

    def _encode_single(self, source):
        return tf.keras.preprocessing.sequence.pad_sequences([
            [
                self.vocab.item2id(char) \
                for char in self.property_selector(word)
            ] \
            for word in source.words
        ], maxlen=self.max_length, dtype=self.dtype, padding='post', truncating='post', value=utils.vocab.PAD_ID)

    def _encode_batch(self, source):
        return tf.keras.preprocessing.sequence.pad_sequences([
            [
                self._encode_word(word) \
                for word in sent.words
            ] \
            for sent in source
        ], dtype=self.dtype, padding='post', truncating='post', value=utils.vocab.PAD_ID)

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

class HeadPropEncoder(BasePropEncoder):
    def __init__(self):
        super(HeadPropEncoder, self).__init__(lambda x: x.head)

        self.dtype = np.uint8

    def _encode_single(self, source):
        return np.fromiter((
            int(self.property_selector(word)) \
            for word in source.words
        ), dtype=self.dtype)

    def _encode_batch(self, source):
        return tf.keras.preprocessing.sequence.pad_sequences([
            [
                int(self.property_selector(word)) \
                for word in sent.words
            ] \
            for sent in source
        ], dtype=self.dtype, padding='post', truncating='post', value=utils.vocab.PAD_ID)

    def encode_onehot(self, source, is_batch):
        y = self.encode(source, is_batch)
        num_classes = y[0].shape[0] if is_batch else len(source)

        return tf.keras.utils.to_categorical(y, num_classes, dtype=self.dtype)
