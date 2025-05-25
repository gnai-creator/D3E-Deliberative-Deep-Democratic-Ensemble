# ClippyX_internal_models.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class ClippyInternalModels:
    def __init__(self):
        self.votos_memoria = []

        # RNN simples para rastrear padrões temporais
        self.rnn = models.Sequential([
            layers.Input(shape=(None, 30 * 30)),
            layers.SimpleRNN(64, return_sequences=False),
            layers.Dense(1, activation="sigmoid")
        ])
        self.rnn.compile(optimizer='adam', loss='binary_crossentropy')

        # AutoEncoder para detectar anomalias
        self.encoder = models.Sequential([
            layers.Input(shape=(30 * 30,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu')
        ])
        self.decoder = models.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(30 * 30, activation='sigmoid')
        ])
        self.autoencoder = models.Sequential([self.encoder, self.decoder])
        self.autoencoder.compile(optimizer='adam', loss='mse')

        # "BNN" fictício como placeholder
        self.fake_bnn_weights = []

    def adicionar_voto(self, voto, consenso):
        self.votos_memoria.append((voto.flatten(), consenso))
        if len(self.votos_memoria) > 100:
            self.votos_memoria = self.votos_memoria[-100:]

    def treinar_todos(self):
        if len(self.votos_memoria) < 10:
            return

        X, y = zip(*self.votos_memoria)
        X = np.array(X)
        y = np.array(y)

        # RNN precisa de sequência
        X_rnn = np.expand_dims(X, axis=0)  # (1, timesteps, features)
        self.rnn.fit(X_rnn, np.mean(y), epochs=3, verbose=0)

        # AutoEncoder
        self.autoencoder.fit(X, X, epochs=5, verbose=0)

        # "BNN" fake (simulando variância como incerteza)
        self.fake_bnn_weights = [np.var(x) for x in X]

    def avaliar_confiança(self, voto):
        flat = voto.flatten()[None, :]
        encoded = self.encoder(flat)
        decoded = self.decoder(encoded)
        ae_loss = tf.reduce_mean(tf.square(flat - decoded)).numpy()

        # RNN
        rnn_pred = self.rnn.predict(np.expand_dims(flat, axis=0), verbose=0)[0][0]

        # BNN fake
        variancia_media = np.mean(self.fake_bnn_weights) if self.fake_bnn_weights else 1.0
        bnn_conf = float(np.exp(-np.var(flat) / (variancia_media + 1e-5)))

        return {
            "rnn": rnn_pred,
            "autoencoder_loss": ae_loss,
            "bnn": bnn_conf
        }
