from utils.encoders import PropVocabEncoder, PropIterVocabEncoder, HeadPropEncoder, FeatsPropEncoder
import conll.vocab

F_FORM = "FORM"
F_FORM_CHAR = "FORM_CHAR"
F_LEMMA_CHAR = "LEMMA_CHAR"
F_UPOS = "UPOS"
F_DEPREL = "DEPREL"
F_FEATS = "FEATS"
F_HEAD = "HEAD"

F_ALL = [F_FORM, F_FORM_CHAR, F_LEMMA_CHAR, F_UPOS, F_FEATS, F_HEAD, F_DEPREL]

class FeaturesEncoder:
    def __init__(self, vocabs, args, x_feats = [F_FORM, F_FORM_CHAR], y_feats = [F_LEMMA_CHAR, F_UPOS, F_FEATS, F_HEAD, F_DEPREL]):
        _ = dict()

        _[F_FORM] = PropVocabEncoder(lambda w: w.form, vocabs[conll.vocab.WORD])
        _[F_FORM_CHAR] = PropIterVocabEncoder(lambda w: w.form, vocabs[conll.vocab.CHAR], args.model_word_max_length)
        _[F_LEMMA_CHAR] = PropIterVocabEncoder(lambda w: w.lemma, vocabs[conll.vocab.CHAR], args.model_word_max_length, onehot=True)
        _[F_UPOS] = PropVocabEncoder(lambda w: w.upos, vocabs[conll.vocab.UPOS], onehot=True)
        _[F_DEPREL] = PropVocabEncoder(lambda w: w.deprel, vocabs[conll.vocab.DEPREL], onehot=True)
        _[F_FEATS] = FeatsPropEncoder(vocabs[conll.vocab.FEATS])
        _[F_HEAD] = HeadPropEncoder(onehot=True)

        self.x_encoders = [_[f] for f in x_feats]
        self.y_encoders = [_[f] for f in y_feats]

    def encode_batch(self, batch):
        return (
            [encoder.encode_batch(batch) for encoder in self.x_encoders],
            [encoder.encode_batch(batch) for encoder in self.y_encoders]
        )
