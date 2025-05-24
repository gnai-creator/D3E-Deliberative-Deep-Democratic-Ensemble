import tensorflow as tf
import numpy as np
from runtime_utils import log
import matplotlib.pyplot as plt
import seaborn as sns
import os

def salvar_voto_visual(votos, iteracao, saida_dir="votos_visuais"):
    os.makedirs(saida_dir, exist_ok=True)
    num_modelos = len(votos)
    votos_classes = [tf.argmax(v, axis=-1).numpy()[0] for v in votos]  # [H, W] para cada

    # Calcular mapa de consenso por pixel
    votos_stack = np.stack(votos_classes, axis=0)  # [M, H, W]
    H, W = votos_stack.shape[1:]

    consenso_map = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            _, counts = np.unique(votos_stack[:, i, j], return_counts=True)
            if np.max(counts) >= 3:
                consenso_map[i, j] = 1  # estabilidade jurídica

    fig, axes = plt.subplots(1, num_modelos + 1, figsize=(4 * (num_modelos + 1), 4))
    nomes = [f"Jurada {i+1}" for i in range(num_modelos - 2)] + ["Advogada", "Juíza"]

    for i, (ax, voto, nome) in enumerate(zip(axes[:-1], votos_classes, nomes)):
        sns.heatmap(voto, ax=ax, cbar=False, cmap="viridis", square=True)
        ax.set_title(f"{nome}", fontsize=10)
        ax.axis("off")

    # Último eixo: consenso
    ax_consenso = axes[-1]
    sns.heatmap(consenso_map, ax=ax_consenso, cbar=False, cmap="Greens", square=True)
    ax_consenso.set_title("Mapa de Consenso (≥3)", fontsize=10)
    ax_consenso.axis("off")

    plt.suptitle(f"Predições dos Modelos - Iteração {iteracao}", fontsize=12)
    filepath = os.path.join(saida_dir, f"votos_iter_{iteracao:02d}.png")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    log(f"[VISUAL] Salvo mapa de votos + consenso em {filepath}")


def arc_court(models, input_tensor, max_iters=5, tol=0.98, epochs=3):
    if len(models) < 5:
        raise ValueError("Corte incompleta: recebi menos de 5 modelos.")

    juradas = [models[i] for i in range(3)]
    advogada = models[3]
    juiza = models[4]

    consenso = 0.0
    iter_count = 0
    votos_final = None

    log(f"[INÍCIO] Tribunal iniciado com {len(models)} modelos.")
    log(f"[INFO] Tolerância de consenso definida em {tol:.2f}")

    while consenso < 1.0 and iter_count < max_iters:
        log(f"\n[ITER {iter_count+1}] Iniciando rodada de julgamento")

        # 1. Advogada faz predição
        y_advogada_logits = advogada(input_tensor, training=False)
        y_advogada_classes = tf.argmax(y_advogada_logits, axis=-1)  # [B, H, W]
        log(f"[INFO] Advogada previu classes com shape: {y_advogada_classes.shape}")

        # 2. Juradas aprendem com a advogada
        for idx, jurada in enumerate(juradas):
            jurada.fit(x=input_tensor, y=y_advogada_classes, epochs=epochs, verbose=0)
            log(f"[TREINO] Jurada {idx+1} treinada com saída da advogada")

        # 3. Juradas produzem suas predições
        saidas_juradas = [jurada(input_tensor, training=False) for jurada in juradas]
        log(f"[INFO] Juradas emitiram opiniões, cada uma com shape: {saidas_juradas[0].shape}")

        # 4. Juíza aprende com concatenação das saídas (juradas + advogada)
        input_juiza = tf.concat(saidas_juradas + [y_advogada_logits], axis=-1)  # [B, H, W, C * 4]
        juiza.fit(x=input_juiza, y=y_advogada_classes, epochs=epochs, verbose=0)
        log(f"[TREINO] Juíza treinada com opiniões de juradas e advogada")

        # 5. Todos votam
        votos_models = [model(input_tensor, training=False) for model in juradas + [advogada]]
        entrada_juiza_final = tf.concat(votos_models, axis=-1)
        voto_juiza = juiza(entrada_juiza_final, training=False)
        votos_models.append(voto_juiza)

        # 6. Plotar Votos
        salvar_voto_visual(votos_models, iter_count)

        consenso = avaliar_consenso_por_j(votos_models, tol)
        log(f"[CONSENSO] Iteração {iter_count+1}: Consenso = {consenso:.4f}")

        iter_count += 1
        votos_final = tf.argmax(votos_models[-1], axis=-1)

    log(f"\n[FIM] Julgamento encerrado após {iter_count} iteração(ões). Consenso final: {consenso:.4f}")
    return votos_final


def avaliar_consenso_por_j(votos_models, tol=0.98):
    """
    Avalia se pelo menos 3 modelos concordam na predição de classe por pixel.
    """
    votos_classe = [tf.argmax(v, axis=-1) for v in votos_models]  # lista de [B, H, W]
    votos_stacked = tf.stack(votos_classe, axis=0)  # [num_modelos, B, H, W]

    def contar_consenso(votos_pixel):
        uniques, _, count = tf.unique_with_counts(votos_pixel)
        return tf.reduce_max(count)

    votos_majoritarios = tf.map_fn(
        lambda x: tf.map_fn(
            lambda y: tf.map_fn(contar_consenso, y, dtype=tf.int32),
            x,
            dtype=tf.int32
        ),
        tf.transpose(votos_stacked, [1, 2, 3, 0]),  # [B, H, W, num_modelos]
        dtype=tf.int32
    )

    consenso_bin = tf.cast(votos_majoritarios >= 3, tf.float32)
    return tf.reduce_mean(consenso_bin).numpy()
