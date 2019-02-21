from utils.vocab import *
import gzip
import numpy as np

class Embeddings:
    def __init__(self, size: int, dim: int, vocab: Vocab, vectors):
        self.size = size
        self.dim = dim
        self.vocab = vocab
        self.vectors = vectors

    @staticmethod
    def from_file(filename, encoding='utf-8', load_vectors=True):

        def _with_vectors(f, size, dim):
            vocab = dict()
            vectors = np.zeros(shape=(size, dim), dtype=np.float32)

            i = len(VOCAB_PREFIX)
            for line in f:
                fields = line.split()
                vocab[fields[0]] = i
                vectors[i,:] = np.fromiter((float(x) for x in fields[1:]), dtype=np.float32)
                i = i + 1

            return Embeddings(size, dim, vocab, vectors)

        def _without_vectors(f, size, dim):
            vocab = dict()
            vectors = None

            i = len(VOCAB_PREFIX)
            for line in f:
                fields = line.split()
                vocab[fields[0]] = i
                i = i + 1

            return Embeddings(size, dim, vocab, vectors)

        f = gzip.open(filename, mode='rt', encoding=encoding) \
            if filename.endswith('.gz') else \
            open(filename, mode='rt', encoding=encoding)

        # header of a file specifies size of a dictionary and dimension of word embedding
        size, dim = map(int, f.readline().strip().split())
        size = size + len(VOCAB_PREFIX)

        return _with_vectors(f, size, dim) if load_vectors else _without_vectors(f, size, dim)
