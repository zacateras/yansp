from utils.encoders import PropVocabEncoder, PropIterVocabEncoder, HeadPropEncoder, FeatsPropEncoder
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
    
    def decode_batch(self, batch: Batch):
        return Batch(
            sents = batch.sents,
            x = None,
            y = dict((f, encoder.decode_batch(batch.y[f])) for (f, encoder) in self.y_encoders)
        )
