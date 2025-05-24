import tensorflow as tf

def arc_court(models, input_tensor, max_iters=5, tol=0.98, epochs=3):
    if len(models) < 5:
        raise ValueError("Corte incompleta: recebi menos de 5 modelos.")

    juradas = [models[i] for i in range(3)]
    advogada = models[3]
    juiza = models[4]

    consenso = 0.0
    iter_count = 0
    votos_final = None

    while consenso < 1.0 : #and iter_count < max_iters:
        # 1. Advogada faz predição [B, W, H, J, T, C]
        y_advogada_logits = advogada(input_tensor, training=False)
        y_advogada_classes = tf.argmax(y_advogada_logits, axis=-1)  # [B, W, H]

        # 2. Juradas aprendem com a advogada
        for jurada in juradas:
            jurada.fit(x=input_tensor, y=y_advogada_classes, epochs=epochs, verbose=0)

        # 3. Juradas produzem suas predições
        saidas_juradas = [jurada(input_tensor, training=False) for jurada in juradas]

        # 4. Juíza aprende com a concatenação das predições das juradas
        input_juiza = tf.concat(saidas_juradas, axis=-1)
        input_juiza = tf.expand_dims(input_juiza, axis=4)
        juiza.fit(x=input_juiza, y=y_advogada_classes, epochs=epochs, verbose=0)

        # 5. Todos votam
        votos_models = [model(input_tensor, training=False) for model in juradas + [advogada, juiza]]
        consenso = avaliar_consenso_por_j(votos_models, tol)

        iter_count += 1
        votos_final = votos_models[-1]  # último voto do juiz é o final

    return votos_final


def avaliar_consenso_por_j(votos_models, tol=0.98):
    """
    Recebe lista de tensores [B, W, H, J, T, C] e avalia se pelo menos 3 modelos
    têm J > tol para cada pixel.
    """
    votos_binarios = [
        tf.cast(tf.reduce_max(model[:, :, :, :, :, :], axis=[3, 4, 5]) > tol, tf.int32)
        for model in votos_models
    ]
    votos_somados = tf.add_n(votos_binarios)  # shape: [B, W, H]
    votos_consenso = tf.reduce_mean(tf.cast(votos_somados >= 3, tf.float32))  # média de pixels com consenso
    return votos_consenso.numpy()