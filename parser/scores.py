import stringdist

class Accuracy:
    def __init__(self, word_score):
        self._word_score = word_score

    def __call__(self, sents_gold, sents_system):
        total = 0.0
        score = 0.0

        for sent_gold, sent_system in zip(sents_gold, sents_system):
            for word_gold, word_system in zip(sent_gold, sent_system):
                total += 1.0
                score += self._word_score(word_gold, word_system)

        return score / total

def levenshtein_norm(gold, system):
    return stringdist.levenshtein_norm(gold.lemma, system.lemma)

class Eq:
    def __init__(self, selector):
        self.selector = selector

    def __call__(self, gold, system):
        return self.selector(gold) == self.selector(system)

def feats(gold, system):
    gold_feats = gold.feats
    system_feats = system.feats

    total = len(gold_feats)
    tp = sum(1 for x in system_feats if x in gold_feats)
    fp = sum(1 for x in system_feats if x not in gold_feats)

    return tp / total / (1.0 + fp)

y = {
    'LEMMA_ACC': Accuracy(levenshtein_norm),
    'UPOS_ACC': Accuracy(Eq(lambda x: x.upos)),
    'FEATS_ACC': Accuracy(feats),
    'HEAD_ACC': Accuracy(Eq(lambda x: x.head)),
    'DEPREL_ACC': Accuracy(Eq(lambda x: x.deprel)),
}
