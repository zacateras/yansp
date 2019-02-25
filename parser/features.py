from utils.encoders import PropVocabEncoder, PropIterVocabEncoder, HeadPropEncoder
import conll.vocab

F_FORM = "FORM"
F_FORM_CHAR = "FORM_CHAR"
F_LEMMA = "LEMMA"
F_LEMMA_CHAR = "LEMMA_CHAR"
F_UPOS = "UPOS"
F_DEPREL = "DEPREL"
F_FEATS = "FEATS"
F_HEAD = "HEAD"

F_ALL = [F_FORM, F_FORM_CHAR, F_LEMMA, F_LEMMA_CHAR, F_UPOS, F_DEPREL, F_FEATS, F_HEAD]

class FeaturesEncoder:
    def __init__(self, vocabs, args):
        _ = dict()

        _[F_FORM] = PropVocabEncoder(lambda w: w.form, vocabs[conll.vocab.WORD]).encode
        _[F_FORM_CHAR] = PropIterVocabEncoder(lambda w: w.form, vocabs[conll.vocab.CHAR], args.model_word_max_length).encode
        _[F_LEMMA] = PropVocabEncoder(lambda w: w.lemma, vocabs[conll.vocab.LEMMA]).encode_onehot
        _[F_LEMMA_CHAR] = PropIterVocabEncoder(lambda w: w.lemma, vocabs[conll.vocab.CHAR], args.model_word_max_length).encode_onehot
        _[F_UPOS] = PropVocabEncoder(lambda w: w.upos, vocabs[conll.vocab.UPOS]).encode_onehot
        _[F_DEPREL] = PropVocabEncoder(lambda w: w.deprel, vocabs[conll.vocab.DEPREL]).encode_onehot
        _[F_FEATS] = PropIterVocabEncoder(lambda w: w.feats, vocabs[conll.vocab.FEATS], args.model_feats_max_length).encode_onehot
        _[F_HEAD] = HeadPropEncoder().encode_onehot

        self.encoders = _

    def encode_batch(self, sents, features = F_ALL):
        encoders = ((F, self.encoders[F]) for F in F_ALL)

        batch = dict()
        for F, encoder in encoders:
            batch[F] = encoder(sents, is_batch=True)

        return batch
