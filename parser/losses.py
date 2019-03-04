import tensorflow as tf
import keras
from parser.features import F_LEMMA_CHAR, F_UPOS, F_FEATS, F_HEAD, F_DEPREL
from keras import backend as K

categorical_crossentropy = keras.losses.categorical_crossentropy

class CycleLoss:
    def __init__(
        self,
        loss_cycle_n: int,
        batch_size: int):

        self.loss_cycle_n = loss_cycle_n
        self.batch_size = batch_size

    def __call__(self, y_true, y_pred):
        loss = 0.0
        if self.loss_cycle_n == 0:
            return loss

        yn = y_pred[:, 1:, 1:]
        for _ in range(self.loss_cycle_n):
            loss += K.sum(tf.trace(yn)) / self.batch_size
            yn = K.batch_dot(yn, y_pred[:, 1:, 1:])

        return loss

class HeadLoss:
    def __init__(
        self,
        loss_cycle_weight: float,
        loss_cycle_n: int,
        batch_size: int):

        self.loss_cycle_weight = loss_cycle_weight
        self.loss_cycle = CycleLoss(loss_cycle_n, batch_size)

    def __call__(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred)
        loss += self.loss_cycle_weight * self.loss_cycle(y_true, y_pred)
        loss = K.mean(loss)

        return loss

class CategoricalCrossentropyLoss:
    def __call__(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred)
        loss = K.mean(loss)

        return loss

class CategoricalCrossentropyFromLogitsLoss:
    def __call__(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss = K.mean(loss)

        return loss

y = lambda args: {
    F_LEMMA_CHAR: CategoricalCrossentropyLoss(),
    F_UPOS: CategoricalCrossentropyLoss(),
    F_FEATS: CategoricalCrossentropyFromLogitsLoss(),
    F_HEAD: HeadLoss(args.loss_cycle_weight, args.loss_cycle_n, args.batch_size),
    F_DEPREL: CategoricalCrossentropyLoss(),
}
