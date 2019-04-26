from collections import Counter

import os
import pickle

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
        self._id2item = dict(zip(item2id.values(), item2id.keys()))

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

    def id2item(self, id):
        if id in self._id2item:
            return self._id2item[id]
        else:
            return UNK

    @staticmethod
    def from_iterable(iterable):
        itemList = [*VOCAB_PREFIX, *iterable]

        item2cnt = Counter(itemList)
        item2id = dict(zip(item2cnt, range(len(item2cnt))))

        return Vocab(item2id, item2cnt)

    @staticmethod
    def from_dict(dict_: dict):
        for value in dict_.values():
            assert isinstance(value, int)
            assert value > len(VOCAB_PREFIX) - 1

        itemList = [*VOCAB_PREFIX, *dict_.keys()]
        idList = [*range(len(VOCAB_PREFIX)), *dict_.values()]

        item2cnt = Counter(itemList)
        item2id = dict(zip(itemList, idList))

        return Vocab(item2id, item2cnt)

    @staticmethod
    def ensure_saved(base_dir, obj):
        file = os.path.join(base_dir, 'vocab')
        if not os.path.exists(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, 'wb+') as f:
                pickle.dump(obj, f)

    @staticmethod
    def load(base_dir, default=None):
        file = os.path.join(base_dir, 'vocab')
        if os.path.exists(file):
            with open(file, 'rb') as f:
                return pickle.load(f)
        elif default is not None:
            return default
        else:
            raise RuntimeError('Vocabulary file was not found in {}.'.format(base_dir))
