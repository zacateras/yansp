import os
from conll.conll18_ud_eval import ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
import conll.conll18_ud_eval
import conll.vocab
import utils
from typing import List, Mapping
from enum import Enum

class UDWord:
    def __init__(self, columns):
        """
        10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
        """
        self.columns = columns

    def __str__(self):
        return self.form

    def __repr__(self):
        return self.__str__()

    @property
    def id(self):
        return int(self.columns[ID])

    @id.setter
    def id(self, value):
        self.columns[ID] = str(value)

    @property
    def form(self):
        return self.columns[FORM]

    @form.setter
    def form(self, value):
        self.columns[FORM] = value

    @property
    def lemma(self):
        return self.columns[LEMMA]

    @lemma.setter
    def lemma(self, value):
        self.columns[LEMMA] = value

    @property
    def upos(self):
        return self.columns[UPOS]

    @upos.setter
    def upos(self, value):
        self.columns[UPOS] = value

    @property
    def xpos(self):
        return self.columns[XPOS]

    @xpos.setter
    def xpos(self, value):
        self.columns[XPOS] = value

    @property
    def feats(self):
        return self.columns[FEATS].split('|')

    @feats.setter
    def feats(self, value):
        self.columns[FEATS] = '|'.join(value)

    @property
    def head(self):
        return int(self.columns[HEAD])

    @head.setter
    def head(self, value):
        self.columns[HEAD] = str(value)

    @property
    def deprel(self):
        return self.columns[DEPREL]

    @deprel.setter
    def deprel(self, value):
        self.columns[DEPREL] = value

    @property
    def deps(self):
        return self.columns[DEPS]

    @deps.setter
    def deps(self, value):
        self.columns[DEPS] = value

    @property
    def misc(self):
        return self.columns[MISC]

    @misc.setter
    def misc(self, value):
        self.columns[MISC] = value

    @property
    def is_multiword(self):
        return False

class UDRoot(UDWord):
    def __init__(self):
        super(UDRoot, self).__init__(None)

    @property
    def id(self):
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
    def xpos(self):
        return utils.vocab.ROOT

    @property
    def feats(self):
        return []

    @property
    def head(self):
        # CoNLL file words point to ROOT at 0 position
        return 0

    @property
    def deprel(self):
        return utils.vocab.ROOT

    @property
    def deps(self):
        return utils.vocab.ROOT

    @property
    def misc(self):
        return utils.vocab.ROOT

    @property
    def is_multiword(self):
        return False

class CoNLLWord(UDWord):
    def __init__(self, word):
        super(CoNLLWord, self).__init__(word.columns)
        
        self._word = word

class UDSentence:
    def __init__(self, words: List[UDWord]):
        self._words = words

    def __str__(self):
        return ' '.join(map(lambda x: x.form, self._words))

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._words)

    def __getitem__(self, key):
        return self._words[key]

    @property
    def words(self):
        return self._words

    def with_root(self):
        return UDSentence([UDRoot()] + self._words)

    @staticmethod  
    def from_UDRepresentation(tb):
        sents = []
        last: int = 0
        words: List[UDWord] = []

        for word in tb.words:
            word = CoNLLWord(word)

            if word.id < last:
                sents.append(UDSentence(words))
                words = []
            
            last = word.id
            words.append(word)

        return sents
            
class CoNLLFile:
    def __init__(self, name, sents, vocabs, lang=None, tag=None, dataset_type=None):
        self._name = name
        self._sents = sents
        self._vocabs = vocabs
        self._lang = lang
        self._tag = tag
        self._dataset_type = dataset_type

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'ud_treebank, {}, {}, {}'.format(self._lang, self._tag, self._dataset_type)

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

    @property
    def dataset_type(self):
        return self._dataset_type

def write_conllu(file, sents: List[UDSentence]):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w+', encoding='utf-8') as f:
        for sent in sents:
            # foreach word except root
            for word in sent.words[1:]:
                f.write('\t'.join(str(column) for column in word.columns) + '\n')

            f.write('\n')

def load_conllu(file, is_path=True, name=None, lang=None, tag=None, dataset_type=None):
    UDR = _just_load_conllu(file, is_path)

    if is_path:
        if name is None:
            name = os.path.basename(file)

            lang, tag = name.split('-')[0].split('_')
            dataset_type = name.split('-')[2].split('.')[0]

    vocabs = conll.vocab.from_UDRepresentation(UDR)
    sents = UDSentence.from_UDRepresentation(UDR)

    return CoNLLFile(name, sents, vocabs, lang=lang, tag=tag, dataset_type=dataset_type)

def evaluate(gold_ud, system_ud):
    gold_ud = _just_load_conllu(gold_ud)
    system_ud = _just_load_conllu(system_ud)

    return conll.conll18_ud_eval.evaluate(gold_ud, system_ud)

def _just_load_conllu(file, is_path=True):
    file = str(file)

    if  is_path:
        assert os.path.exists(file)
        assert file.endswith('.conllu')

        with open(file) as f:
            return conll.conll18_ud_eval.load_conllu(f)

    else:
        return conll.conll18_ud_eval.load_conllu(file)
