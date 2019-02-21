from collections import Counter

PAD = '<PAD>'
PAD_ID = 0
UNK = '<UNK>'
UNK_ID = 1
EMPTY = '<EMPTY>'
EMPTY_ID = 2
ROOT = '<ROOT>'
ROOT_ID = 3
VOCAB_PREFIX = [PAD, UNK, EMPTY, ROOT]

class Vocab:
    def __init__(self, item2id, item2cnt):
        self._item2id = item2id
        self._item2cnt = item2cnt

    def __str__(self):
        return self._item2id.__str__()

    def __repr__(self):
        return self._item2id.__repr__()

    def __len__(self):
        return len(self._item2id)

    def __getitem__(self, key):
        return self._item2id[key]
        
    @property
    def size(self) -> int:
        return len(self)

    def item2id(self, item):
        if item in self._item2id:
            return self._item2id[item]
        else:
            return self._item2id[UNK]

    @staticmethod
    def from_iterable(iterable):
        item2cnt = Counter(VOCAB_PREFIX + list(iterable))
        item2id = dict(zip(item2cnt, range(len(item2cnt))))

        return Vocab(item2id, item2cnt)

    @staticmethod
    def from_dict(dict_: dict):
        for value in dict_.values():
            assert isinstance(value, int)
            assert value < len(VOCAB_PREFIX)

        item2cnt = Counter(VOCAB_PREFIX + dict_.keys())
        item2id = dict(zip(VOCAB_PREFIX + dict_.keys(), range(len(VOCAB_PREFIX)) + dict_.values()))

        return Vocab(item2id, item2Cnt)
