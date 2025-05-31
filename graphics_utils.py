import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from metrics_utils import garantir_dict_votos_models
from runtime_utils import log
from metrics_utils import extrair_matriz_simbolica
from metrics_utils import extrair_matriz_simbolica_test
import scipy.stats  # Importa o pacote completo para uso robusto

def salvar_voto_visual(votos, iteracao, block_idx, input_tensor_outros, classes_validas, classes_objetivo, consenso, idx=0, task_id=None, saida_dir="debug_plots", filename="a"):
    os.makedirs(saida_dir, exist_ok=True)
    fname = filename
    filepath = os.path.join(saida_dir, fname)

    votos_classes = []
    softmax_maxes = []
    classes_resumidas = []

    for i, (nome, voto) in enumerate(garantir_dict_votos_models(votos).items()):
        if voto is None:
            continue
        try:
            if isinstance(voto, tf.Tensor):
                voto = voto.numpy()

            if voto.ndim == 5:
                voto = voto[0]  # remove batch dim

            if voto.ndim == 4 and voto.shape == (1, 30, 30, 3):
                voto = voto[0]  # remove batch dim

            if voto.ndim == 4 and voto.shape[-1] == 10:
                v_soft = tf.nn.softmax(voto.astype(np.float32), axis=-1).numpy()
                v_cls = np.argmax(v_soft, axis=-1).astype(np.int32)
                softmax_maxes.append(np.max(v_soft, axis=-1))

            elif voto.ndim == 3:
                if voto.shape[-1] == 1:
                    v_cls = np.squeeze(voto, axis=-1).astype(np.int32)
                    v_cls[v_cls == -1] = 0  # evita classe -1
                    softmax_maxes.append(np.zeros_like(v_cls))
                    log(f"[VISUAL DEBUG] Voto {i} interpretado como simbólico com shape (30,30,1).")
                elif voto.shape[-1] == 10 and np.any(voto > 1):
                    v_soft = tf.nn.softmax(voto.astype(np.float32), axis=-1).numpy()
                    v_cls = np.argmax(v_soft, axis=-1).astype(np.int32)
                    softmax_maxes.append(np.max(v_soft, axis=-1))
                elif voto.shape[-1] == 3:
                    float_block = voto[:, :, 0].astype(np.float32)
                    rounded = np.floor(float_block + 0.5)
                    clipped = np.clip(rounded, 0, 9)
                    v_cls = clipped.astype(np.int32)
                    v_cls[v_cls == -1] = 0  # evita classe -1
                    softmax_maxes.append(np.zeros_like(v_cls))
                else:
                    v_cls = np.argmax(voto, axis=-1).astype(np.int32)
                    softmax_maxes.append(np.zeros_like(v_cls))

            elif voto.ndim == 2:
                v_cls = voto.astype(np.int32)
                softmax_maxes.append(np.zeros_like(v_cls))
            else:
                raise ValueError(f"[VISUAL DEBUG] Formato de voto não suportado: {voto.shape}")

            if i == 6:
                v_cls = 9 - v_cls

            votos_classes.append(v_cls)
            classes_resumidas.append(np.unique(v_cls))

        except Exception as e:
            print(f"[VISUAL DEBUG] Erro ao preparar voto do modelo_{i}: {e}")

    if not votos_classes:
        print("[VISUAL DEBUG] ❌ Nenhuma predição válida para visualização.")
        return

    input_vis = input_tensor_outros
    if isinstance(input_vis, tf.Tensor):
        input_vis = input_vis.numpy()
    if input_vis.ndim == 5:
        input_vis = input_vis[0]
    if input_vis.ndim == 4 and input_vis.shape[-1] == 10:
        input_vis = np.argmax(input_vis, axis=-1).astype(np.int32)
    elif input_vis.ndim == 3:
        if input_vis.shape[-1] == 10:
            input_vis = np.argmax(input_vis, axis=-1).astype(np.int32)
        elif input_vis.shape[-1] == 3:
            float_block = input_vis[:, :, 0].astype(np.float32)
            rounded = np.floor(float_block + 0.5)
            clipped = np.clip(rounded, 0, 9)
            input_vis = clipped.astype(np.int32)
        elif input_vis.shape[-1] == 1:
            input_vis = input_vis[:, :, 0].astype(np.int32)
        elif input_vis.shape[-1] > 1:
            input_vis = np.argmax(input_vis, axis=-1).astype(np.int32)
    elif input_vis.ndim == 2:
        input_vis = input_vis.astype(np.int32)
    else:
        input_vis = np.zeros((30, 30), dtype=np.int32)

    votos_stack = np.stack(votos_classes, axis=0)
    h, w = votos_stack.shape[1:3]
    entropia_map = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            _, counts = np.unique(votos_stack[:, i, j], return_counts=True)
            probs = counts / counts.sum()
            entropia_map[i, j] = scipy.stats.entropy(probs, base=2)

    num_modelos = len(votos_classes)
    fig, axes = plt.subplots(2, num_modelos + 1, figsize=(4 * (num_modelos + 1), 8))
    cargos = {
        0: "Jurada 1", 1: "Jurada 2", 2: "Jurada 3",
        3: "Advogada", 4: "Juíza", 5: "Suprema Juíza", 6: "Promotor"
    }

    for i in range(num_modelos):
        nome = cargos.get(i, f"Modelo {i}")
        voto = votos_classes[i]
        smap = softmax_maxes[i]
        classes_int = classes_resumidas[i].astype(np.int32)

        classes_formatadas = ", ".join(map(str, classes_int.tolist()))

        axes[0, i].imshow(voto.astype(np.int32), cmap="viridis", vmin=0, vmax=9, interpolation="nearest")
        axes[0, i].set_title(f"{nome}\nClasses: [{classes_formatadas}]")
        axes[0, i].axis("off")

        axes[1, i].imshow(smap, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        axes[1, i].set_title("Confiança Máx.")
        axes[1, i].axis("off")

    axes[0, -1].imshow(input_vis.astype(np.int32), cmap="viridis", vmin=0, vmax=9)
    axes[0, -1].set_title("Input")
    axes[0, -1].axis("off")

    axes[1, -1].imshow(entropia_map, cmap="inferno", vmin=0, vmax=np.log2(num_modelos))
    axes[1, -1].set_title("Entropia")
    axes[1, -1].axis("off")

    plt.suptitle(f"Task {task_id} — Iteração {iteracao} — Bloco {block_idx}\n Classes Válidas — {classes_validas} — Classes Objetivo — {classes_objetivo}\n Consenso: {consenso} ", fontsize=14)
    plt.tight_layout()
    print(f"[VISUAL DEBUG] Salvando figura em {filepath} — modelos plotados: {num_modelos}")
    plt.savefig(filepath)
    plt.close()
    print(f"[VISUAL DEBUG] ✅ Voto visual detalhado salvo em {filepath}")


