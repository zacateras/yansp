import conll.conll18_ud_eval as conll
from enum import Enum
from utils.vocab import Vocab

CHAR = 101
WORD = 102
FORM = conll.FORM
LEMMA = conll.LEMMA
UPOS = conll.UPOS
XPOS = conll.XPOS
FEATS = conll.FEATS
DEPREL = conll.DEPREL

def from_UDRepresentation(tb):
    flatten = lambda l: [item for sublist in l for item in sublist]

    vocabs = dict()

    vocabs[CHAR] = Vocab.from_iterable(tb.characters)
    vocabs[FORM] = Vocab.from_iterable(map(lambda w: w.columns[conll.FORM], tb.words))
    vocabs[LEMMA] = Vocab.from_iterable(map(lambda w: w.columns[conll.LEMMA], tb.words))
    vocabs[UPOS] = Vocab.from_iterable(map(lambda w: w.columns[conll.UPOS], tb.words))
    vocabs[XPOS] = Vocab.from_iterable(map(lambda w: w.columns[conll.XPOS], tb.words))
    vocabs[FEATS] = Vocab.from_iterable(flatten(map(lambda w: w.columns[conll.FEATS].split('|'), tb.words)))
    vocabs[DEPREL] = Vocab.from_iterable(map(lambda w: w.columns[conll.DEPREL], tb.words))

    return vocabs
