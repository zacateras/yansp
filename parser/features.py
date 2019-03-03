from utils.encoders import PropVocabEncoder, PropIterVocabEncoder, HeadPropEncoder, FeatsPropEncoder

import conll
import conll.vocab

F_FORM = "FORM"
F_FORM_CHAR = "FORM_CHAR"
F_LEMMA_CHAR = "LEMMA_CHAR"
F_UPOS = "UPOS"
F_DEPREL = "DEPREL"
F_FEATS = "FEATS"
F_HEAD = "HEAD"

F_X = [
    F_FORM,
    F_FORM_CHAR
]

F_Y = [
    F_LEMMA_CHAR,
    F_UPOS,
    F_FEATS,
    F_HEAD,
    F_DEPREL
]

F_ALL = [F_FORM, F_FORM_CHAR, F_LEMMA_CHAR, F_UPOS, F_FEATS, F_HEAD, F_DEPREL]

class Batch:
    def __init__(self, sents, x, y):
        self.sents = sents
        self.x = x
        self.y = y

class FeaturesEncoder:
    def __init__(self, vocabs, args, x_feats = F_X, y_feats = F_Y):
        _ = dict()

        _[F_FORM] = PropVocabEncoder(lambda w: w.form, vocabs[conll.vocab.WORD])
        _[F_FORM_CHAR] = PropIterVocabEncoder(lambda w: w.form, vocabs[conll.vocab.CHAR], args.model_word_max_length)
        _[F_LEMMA_CHAR] = PropIterVocabEncoder(lambda w: w.lemma, vocabs[conll.vocab.CHAR], args.model_word_max_length, onehot=True)
        _[F_UPOS] = PropVocabEncoder(lambda w: w.upos, vocabs[conll.vocab.UPOS], onehot=True)
        _[F_DEPREL] = PropVocabEncoder(lambda w: w.deprel, vocabs[conll.vocab.DEPREL], onehot=True)
        _[F_FEATS] = FeatsPropEncoder(vocabs[conll.vocab.FEATS])
        _[F_HEAD] = HeadPropEncoder(onehot=True)

        self.x_encoders = [(f, _[f]) for f in x_feats]
        self.y_encoders = [(f, _[f]) for f in y_feats]

    def encode_batch(self, sents):
        return Batch(
            sents = sents,
            x = dict((f, encoder.encode_batch(sents)) for (f, encoder) in self.x_encoders),
            y = dict((f, encoder.encode_batch(sents)) for (f, encoder) in self.y_encoders)
        )
    
    def decode_batch(self, batch: Batch, y=None):
        if y is None:
            y = batch.y

        y = dict((f, encoder.decode_batch(y[f])) for (f, encoder) in self.y_encoders)

        sents = []

        for sent_i in range(len(batch.sents)):
            sent = batch.sents[sent_i]

            words = list()
            for word_i in range(len(sent)):
                columns = [
                    str(sent[word_i].id),                 # 1 id
                    sent[word_i].form,                    # 2 form
                    y[F_LEMMA_CHAR][sent_i][word_i],      # 3 lemma
                    y[F_UPOS][sent_i][word_i],            # 4 upos
                    '_',                                  # 5 xpos
                    '|'.join(y[F_FEATS][sent_i][word_i]), # 6 feats
                    str(y[F_HEAD][sent_i][word_i]),       # 7 head
                    y[F_DEPREL][sent_i][word_i],          # 8 deprel
                    '_',                                  # 9 deps
                    '_'                                   # 10 misc
                ]

                words.append(conll.UDWord(columns))

            sents.append(conll.UDSentence(words))

        return sents
