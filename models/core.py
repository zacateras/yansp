import tensorflow as tf

class CoreModel(tf.keras.Model):
    def __init__(
        self,
        lstm_layers: int,
        lstm_units: int,
        lstm_dropout: float,
        dropout: float,
        noise: float):

        super(CoreModel, self).__init__()

        self.biLSTM = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=lstm_units,
                    dropout=lstm_dropout,
                    recurrent_dropout=lstm_dropout,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(0.000001),
                    bias_regularizer=tf.keras.regularizers.l2(0.000001),
                    recurrent_regularizer=tf.keras.regularizers.l2(0.000001),
                    activity_regularizer=tf.keras.regularizers.l2(0.000001)
                )
            ) \
            for x in range(lstm_layers)
        ]

        # last layers are taken first

        self.dropout = [
            tf.keras.layers.GaussianDropout(dropout) \
            for x in range(lstm_layers + 1) 
        ]

        self.noise = [
            tf.keras.layers.GaussianNoise(noise) \
            for x in range(lstm_layers + 1) 
        ]

    def call(self, inputs):
        x = self.dropout[-1](inputs)
        x = self.noise[-1](x)

        for i in range(len(self.biLSTM)):
            x = self.biLSTM[i](x)
            x = self.dropout[i](x)
            x = self.noise[i](x)

        return x
