import os
from conll.conll18_ud_eval import ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
import conll.conll18_ud_eval
import conll.vocab
import utils
from typing import List, Mapping
from enum import Enum

class UDWord:
    def __str__(self):
        return self.form

    def __repr__(self):
        return self.__str__()

    @property
    def id(self):
        raise NotImplementedError

    @property
    def head(self):
        raise NotImplementedError

    @property
    def form(self):
        raise NotImplementedError

    @property
    def lemma(self):
        raise NotImplementedError

    @property
    def upos(self):
        raise NotImplementedError

    @property
    def feats(self):
        raise NotImplementedError

    @property
    def deprel(self):
        raise NotImplementedError

    @property
    def is_multiword(self):
        raise NotImplementedError

class UDRoot(UDWord):
    @property
    def id(self):
        return utils.vocab.ROOT

    @property
    def head(self):
        # CoNLL file words point to ROOT at 0 position
        return 0

    @property
    def form(self):
        return utils.vocab.ROOT

    @property
    def lemma(self):
        return utils.vocab.ROOT

    @property
    def upos(self):
        return utils.vocab.ROOT

    @property
    def feats(self):
        return utils.vocab.ROOT

    @property
    def deprel(self):
        return utils.vocab.ROOT

    @property
    def is_multiword(self):
        return False


class CoNLLWord(UDWord):
    def __init__(self, word):
        self._word = word

        if word.is_multiword:
            bounds = word.columns[ID].split('-')
            self._start = int(bounds[0])
            self._end = int(bounds[1])
        else:
            self._start = self._end = int(word.columns[ID])

        self._head = int(word.columns[HEAD])

    @property
    def columns(self):
        """
        10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
        """
        return self._word.columns

    @property
    def id(self):
        return self._start

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def head(self):
        return self._head

    @property
    def form(self):
        return self._word.columns[FORM]

    @property
    def lemma(self):
        return self._word.columns[LEMMA]

    @property
    def upos(self):
        return self._word.columns[UPOS]

    @property
    def feats(self):
        return self._word.columns[FEATS].split('|')

    @property
    def deprel(self):
        return self._word.columns[DEPREL]

    @property
    def is_multiword(self):
        """
        is_multiword==True means that this word is part of a multi-word token.
        """
        return self._word.is_multiword

class UDSentence:
    def __init__(self, words: List[UDWord]):
        self._words = words

    def __str__(self):
        return ' '.join(map(lambda x: x.form, self._words))

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._words)

    @property
    def words(self):
        return self._words

    @staticmethod  
    def from_UDRepresentation(tb):
        last: int = 0
        root = UDRoot()
        words: List[UDWord] = [root]

        for word in tb.words:
            word = CoNLLWord(word)

            if word.id < last:
                yield UDSentence(words)
                words = [root]
            
            last = word.id
            words.append(word)
            
class CoNLLFile:
    def __init__(self, name, sents, vocabs, lang=None, tag=None):
        self._name = name
        self._sents = sents
        self._vocabs = vocabs
        self._lang = lang
        self._tag = tag

    @property
    def name(self):
        return self._name
    
    @property
    def sents(self):
        return self._sents

    @property
    def vocabs(self):
        return self._vocabs

    @property
    def lang(self):
        return self._lang

    @property
    def tag(self):
        return self._tag

def load_conllu(file, is_path=True, name=None, lang=None, tag=None):
    file = str(file)

    if  is_path:
        assert os.path.exists(file)
        assert file.endswith('.conllu')

        if name is None:
            name = os.path.basename(file)

        with open(file) as f:
            UDR = conll.conll18_ud_eval.load_conllu(f)

    else:
        UDR = conll.conll18_ud_eval.load_conllu(file)

    if name is not None and lang is None and tag is None:
        lang, tag = name.split('-')[0].split('_')

    vocabs = conll.vocab.from_UDRepresentation(UDR)
    sents = UDSentence.from_UDRepresentation(UDR)

    return CoNLLFile(name, sents, vocabs, lang=lang, tag=tag)
