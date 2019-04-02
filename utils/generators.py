import random
import sys

class BatchGenerator:
    """
    Splits list of items into batches of batch_size length.
    If item_length_batching is set to True batch length is
    determined by the sum of item lengths, otherwise by the
    count of items.
    """
    def __init__(self, items, batch_size, item_length_batching=True):
        self.items = items
        self.batch_size = batch_size
        self.item_length_batching = item_length_batching

    def _iterate_batches(self, items_ordered, batch_size):
        batch_buffer = []

        # split into batches by length of at least batch_size
        batch_size_counter = 0
        for item in items_ordered:
            batch_buffer.append(item)
            batch_size_counter += len(item) if self.item_length_batching else 1

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
    
class LenwiseBatchGenerator(BatchGenerator):
    def __init__(self, items, batch_size, descending=False, randomize_iteration=False):
        # sort by length
        items = sorted(self.items, key=lambda item: len(item), reverse=descending)

        super(LenwiseBatchGenerator, self).__init__(items, batch_size)

        self.batches = self._iterate_batches(self.items, self.batch_size)
        self.randomize_iteration = randomize_iteration

    def __next__(self):
        while True:
            if self.randomize_iteration:
                return random.choice(self.batches)
            else:
                for batch in self.batches:
                    return batch

class RandomBatchGenerator(BatchGenerator):
    def __init__(self, items, batch_size, item_length_batching=True):
        items = list(items)

        super(RandomBatchGenerator, self).__init__(items, batch_size, item_length_batching)

        self.items_ordered = self._random_iter_from_list(self.items)

    def __next__(self):
        while True:
            return next(self._iterate_batches(self.items_ordered, self.batch_size))

    def _random_iter_from_list(self, x):
        while True:
            i = random.randint(0, len(x) - 1)
            yield x[i]

class OneshotBatchGenerator(BatchGenerator):
    def __init__(self, items, batch_size, limit=sys.maxsize, item_length_batching=True):
        super(OneshotBatchGenerator, self).__init__(items, batch_size)

        self.item_length_batching = item_length_batching
        self.remaining = limit if limit is not None else sys.maxsize
        self.batches = self._iterate_batches(self.items, self.batch_size)

    def __next__(self):
        for batch in self.batches:
            if self.remaining <= 0:
                raise StopIteration

            if self.item_length_batching:
                self.remaining -= sum(len(item) for item in batch)
            else:
                self.remaining -= len(batch)
            
            return batch

        raise StopIteration
