import keras

class LayerNorm(keras.layers.Layer):
    """
    Normalization layer from Hinton et. al.: https://arxiv.org/pdf/1607.06450.pdf
    """
    def __init__(
        self,
        eps=1e-6,
        gamma_initializer='ones',
        beta_initializer='zeros',
        gamma_regularizer=None,
        beta_regularizer=None,
        *args, **kwargs):

        super(LayerNorm, self).__init__(*args, **kwargs)
        self.eps = eps
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.gamma = self.add_weight(
            shape=(input_dim),
            initializer=self.gamma_initializer,
            name='gamma',
            regularizer=self.gamma_regularizer)

        self.beta = self.add_weight(
            shape=(input_dim),
            initializer=self.beta_initializer,
            name='beta',
            regularizer=self.beta_regularizer)

        super(LayerNorm, self).build(input_shape)

    def call(self, x):
        mean = keras.backend.mean(x, axis=-1, keepdims=True)
        std = keras.backend.std(x, axis=-1, keepdims=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        assert input_shape
        return input_shape

    def get_config(self):
        config = {
            'eps': self.eps,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.gamma_initializer)
        }
        base_config = super(LayerNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))