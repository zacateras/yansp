import random

class SentBatchGenerator:
    def __init__(self, sents, batch_size):
        self.sents = sents
        self.batch_size = batch_size

    def _iterate_batches(self, sents_ordered, batch_size):
        batch_buffer = []

        # split into batches by word count of at least batch_size
        batch_size_counter = 0
        for sent in sents_ordered:
            batch_buffer.append(sent)
            batch_size_counter += len(sent)

            if batch_size_counter >= batch_size:
                yield batch_buffer
                batch_buffer = []
                batch_size_counter = 0

        # add leftover batch
        if batch_size_counter > 0:
            yield batch_buffer

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError
    
class LenwiseSentBatchGenerator(SentBatchGenerator):
    def __init__(self, sents, batch_size, descending=False, randomize_iteration=False):
        # sort by length
        sents = sorted(self.sents, key=lambda sent: len(sent), reverse=descending)

        super(LenwiseSentBatchGenerator, self).__init__(sents, batch_size)

        self.batches = self._iterate_batches(self.sents, self.batch_size)
        self.randomize_iteration = randomize_iteration

    def __next__(self):
        while True:
            if self.randomize_iteration:
                return random.choice(self.batches)
            else:
                for batch in self.batches:
                    return batch
        

class RandomSentBatchGenerator(SentBatchGenerator):
    def __init__(self, sents, batch_size):
        sents = list(sents)

        super(RandomSentBatchGenerator, self).__init__(sents, batch_size)

        self.sents_ordered = self._random_iter_from_list(self.sents)

    def __next__(self):
        while True:
            return next(self._iterate_batches(self.sents_ordered, self.batch_size))

    def _random_iter_from_list(self, x):
        while True:
            i = random.randint(0, len(x) - 1)
            yield x[i]
    