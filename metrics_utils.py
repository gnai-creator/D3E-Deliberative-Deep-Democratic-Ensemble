# metrics_utils.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from runtime_utils import log
import cv2
import glob
import sys
sys.path.append('/home/vscode/.local/lib/python3.10/site-packages')
import moviepy.editor as mpy
import scipy.stats  # Importa o pacote completo para uso robusto


os.makedirs("images", exist_ok=True)
sns.set(style="whitegrid", font_scale=1.2)

def ensure_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else x

def extrair_matriz_simbolica(grid_3d, pad_value=0):
    return grid_3d[:, :, 0].astype(np.int32)

def extrair_matriz_simbolica_test(grid_3d, pad_value=0):
    if grid_3d.shape[-1] == 10:
        return np.argmax(grid_3d, axis=-1).astype(np.int32)
    return grid_3d[:, :, 0].astype(np.int32)


def garantir_dict_votos_models(votos_models):
    if isinstance(votos_models, dict):
        return votos_models
    elif isinstance(votos_models, list):
        return {f"modelo_{i}": v for i, v in enumerate(votos_models)}
    else:
        log(f"[SECURITY] votos_models tinha tipo inesperado: {type(votos_models)}. Substituindo por dict vazio.")
        return {}

def prepare_display_image(img, pad_value, h, w):
    img = ensure_numpy(img)
    # print(f"[DEBUG] prepare_display_image - input shape: {img.shape}")
    if img.ndim == 5:
        img = img[0, :, :, 0, :]  # [B, H, W, C, J] → [H, W, J]
        return img
    elif img.ndim == 4:
        if img.shape[-1] == 1:
            return img[0, :, :, 0] if img.shape[0] == 1 else img[:, :, 0, 0]
        if img.shape[0] == 1:
            img = img[0]
        return img
    elif img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]  # (1, W, H) → (W, H)
        return img
    elif img.ndim == 2:
        if img.shape[0] == 1:
            return img[0].reshape((h, w, 1))
        return img
    elif img.ndim == 1:
        side = int(np.sqrt(img.shape[0]))
        return img.reshape((side, side))
    else:
        print(f"[DEBUG] prepare_display_image - unhandled shape: {img.shape}")
        return np.full((h, w, 3), pad_value)

def plot_prediction_test(predicted_output, raw_input, pad_value, save_path):
    try:
        print(f"[DEBUG] plot_prediction_test - predicted_output original shape: {getattr(predicted_output, 'shape', 'N/A')}")
        predicted_output = prepare_display_image(predicted_output, pad_value, 30, 30)
        print(f"[DEBUG] plot_prediction_test - predicted_output processed shape: {predicted_output.shape}")

        if raw_input is not None:
            print(f"[DEBUG] plot_prediction_test - raw_input original shape: {getattr(raw_input, 'shape', 'N/A')}")
            raw_input = prepare_display_image(raw_input, pad_value, 30, 30)
            print(f"[DEBUG] plot_prediction_test - raw_input processed shape: {raw_input.shape}")
        else:
            raw_input = np.zeros_like(predicted_output)

        if predicted_output.ndim in [2, 3]:
            predicted_output = np.squeeze(predicted_output).astype(np.int32)
        else:
            raise ValueError(f"Unexpected shape for predicted_output: {predicted_output.shape}")

        h, w = predicted_output.shape[:2] if predicted_output.ndim == 3 else predicted_output.shape

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(raw_input)
        axes[0].set_title("Input")
        axes[1].imshow(predicted_output)
        axes[1].set_title("Prediction")

        for ax in axes:
            ax.axis("off")
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[INFO] Resultado do teste salvo em: {save_path}")
        plt.close()

    except Exception as e:
        print("[ERROR] Falha ao gerar plot de teste:", str(e))



def ensure_numpy(tensor):
    return tensor.numpy() if hasattr(tensor, "numpy") else tensor

def preparar_voto_para_visualizacao(v):
    try:
        # Converte para numpy se necessário
        if isinstance(v, tf.Tensor):
            v = v.numpy()

        # Remove batch dimension
        if v.ndim == 5:
            v = v[0]  # (30, 30, 10, C) → (30, 30, 10, C)
        
        # Caso padrão: logits ou mapa simbólico → extrai classe dominante
        if v.ndim == 4:
            v = extrair_matriz_simbolica_test(v)

        elif v.ndim == 3:
            if v.shape[-1] == 1:
                v = extrair_matriz_simbolica_test(v)
                log(f"[VISUAL] v extraído (shape=3D com 1 canal): {v.shape}")
            else:
                v = v[:, :, 0]
                log(f"[VISUAL] v reduzido manualmente ao canal 0: {v.shape}")

        v = np.squeeze(v)

        if v.shape != (30, 30, 10):
            log(f"[VISUAL] ⚠️ v final com shape inesperado: {v.shape}")

        return v

    except Exception as e:
        log(f"[VISUAL] Erro ao extrair matriz simbólica: {e}")
        return np.zeros((30, 30))

def salvar_voto_visual(votos, iteracao, block_idx, input_tensor_outros, idx=0, task_id=None, saida_dir="debug_plots"):


    os.makedirs(saida_dir, exist_ok=True)
    fname = f"a.png"
    # fname = f"voto_visual_task{task_id}_iter{iteracao}_bloco{block_idx}.png"
    filepath = os.path.join(saida_dir, fname)

    votos_classes = []
    softmax_maxes = []

    for i, (nome, voto) in enumerate(garantir_dict_votos_models(votos).items()):
        if voto is None:
            continue
        try:
            if isinstance(voto, tf.Tensor):
                voto = voto.numpy()

            if voto.ndim == 5:
                voto = voto[0]  # remove batch dim

            if voto.ndim == 4 and voto.shape[-1] == 10:
                v_soft = tf.nn.softmax(voto.astype(np.float32), axis=-1).numpy()
                v_cls = np.argmax(v_soft, axis=-1)
                softmax_maxes.append(np.max(v_soft, axis=-1))

            elif voto.ndim == 3:
                if voto.shape == (30, 30, 10):
                    # imprimir todas as 10 dimensões de ndim=3 dim=-1
                    for d in range(voto.shape[-1]):
                        log(f"VOTO {i}: {voto[:,:, d]}")
                    v_cls = extrair_matriz_simbolica_test(voto).astype(np.int32)
                    softmax_maxes.append(np.zeros_like(v_cls))
                elif voto.shape[-1] == 10 and np.any(voto > 1):
                    v_soft = tf.nn.softmax(voto.astype(np.float32), axis=-1).numpy()
                    v_cls = np.argmax(v_soft, axis=-1)
                    softmax_maxes.append(np.max(v_soft, axis=-1))
                else:
                    v_cls = np.argmax(voto, axis=-1)
                    softmax_maxes.append(np.zeros_like(v_cls))

            elif voto.ndim == 2:
                v_cls = voto.astype(np.int32)
                softmax_maxes.append(np.zeros_like(v_cls))
            else:
                raise ValueError(f"[VISUAL DEBUG] Formato de voto não suportado: {voto.shape}")

            if i == 6:
                v_cls = 9 - v_cls

            votos_classes.append(v_cls)

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
        input_vis = extrair_matriz_simbolica_test(input_vis)
    elif input_vis.ndim == 3:
        if input_vis.shape[-1] == 10:
            input_vis = extrair_matriz_simbolica_test(input_vis)
        elif input_vis.shape[-1] == 1:
            input_vis = input_vis[:, :, 0]
        elif input_vis.shape[-1] > 1:
            input_vis = extrair_matriz_simbolica_test(input_vis)
    elif input_vis.ndim == 2:
        input_vis = input_vis.astype(np.int32)
    else:
        input_vis = np.zeros((30, 30))

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

        if voto.ndim == 3 and voto.shape[-1] == 10:
            voto = extrair_matriz_simbolica_test(voto)
        elif voto.ndim == 3:
            voto = extrair_matriz_simbolica_test(voto)

        axes[0, i].imshow(voto, cmap="viridis", vmin=0, vmax=9, interpolation="nearest")
        axes[0, i].set_title(f"{nome}\nClasses: {np.unique(voto)}")
        axes[0, i].axis("off")

        axes[1, i].imshow(smap, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        axes[1, i].set_title("Confiança Máx.")
        axes[1, i].axis("off")

    axes[0, -1].imshow(input_vis, cmap="viridis", vmin=0, vmax=9)
    axes[0, -1].set_title("Input")
    axes[0, -1].axis("off")

    axes[1, -1].imshow(entropia_map, cmap="inferno", vmin=0, vmax=np.log2(num_modelos))
    axes[1, -1].set_title("Entropia")
    axes[1, -1].axis("off")

    plt.suptitle(f"Task {task_id} — Iteração {iteracao} — Bloco {block_idx}", fontsize=14)
    plt.tight_layout()
    print(f"[VISUAL DEBUG] Salvando figura em {filepath} — modelos plotados: {num_modelos}")
    plt.savefig(filepath)
    plt.close()
    print(f"[VISUAL DEBUG] ✅ Voto visual detalhado salvo em {filepath}")



def ensure_numpy(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def compute_pixelwise_accuracy(expected, predicted, pad_value=0):
    expected = np.asarray(expected)
    predicted = np.asarray(predicted)

    mask = expected != pad_value
    total_pixels = np.sum(mask)
    if total_pixels == 0:
        return 0.0, 0.0

    correct_pixels = np.sum((expected == predicted) & mask)
    pixel_accuracy = correct_pixels / total_pixels

    shape_accuracy = (expected == predicted).mean()

    return pixel_accuracy, shape_accuracy




def plot_prediction_debug(
    raw_input,
    expected_output,
    predicted_output,
    model_index="",
    pad_value=0,
    index=0,
    task_id=None,
    saida_dir="debug_plots"
):
    
    os.makedirs(saida_dir, exist_ok=True)
    fname = f"a.png"
    # fname = f"voto_visual_task{task_id}_iter{iteracao}_bloco{block_idx}.png"
    filepath = os.path.join(saida_dir, fname)

    log(f"[DEBUG] EXPECTED OUTPUT SHAPE: {expected_output.shape}")
    log(f"[DEBUG] PREDICTED OUTPUT SHAPE: {predicted_output.shape}")

    # Normaliza expected_output para (30, 30, 1)
    if isinstance(expected_output, tf.Tensor):
        expected_output = expected_output.numpy()
    expected_output = np.squeeze(expected_output)
    if expected_output.ndim == 2:
        expected_output = np.expand_dims(expected_output, axis=-1)

    # Normaliza predicted_output para (30, 30, 1)
    if isinstance(predicted_output, tf.Tensor):
        predicted_output = predicted_output.numpy()
    predicted_output = np.squeeze(predicted_output)
    if predicted_output.ndim == 2:
        predicted_output = np.expand_dims(predicted_output, axis=-1)

    # Extrai o canal único para visualização
    expected_2d = expected_output[:, :, 0]
    predicted_2d = predicted_output[:, :, 0]

    if expected_2d.shape != predicted_2d.shape:
        log(f"[ERROR] Shape incompatível: predicted={predicted_2d.shape}, expected={expected_2d.shape}")
        return 0, 0

    # Métricas
    mask = expected_2d != pad_value
    total = np.sum(mask)
    correct = np.sum((expected_2d == predicted_2d) & mask)
    pixel_color_perfect = correct / total if total > 0 else 0.0
    pixel_shape_perfect = (expected_2d == predicted_2d).mean()

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(expected_2d, cmap="viridis", vmin=0, vmax=9)
    ax[0].set_title("Expected Output")
    ax[0].axis("off")

    ax[1].imshow(predicted_2d, cmap="viridis", vmin=0, vmax=9)
    ax[1].set_title("Predicted Output")
    ax[1].axis("off")

    plt.suptitle(
        f"Task {task_id} — Model {model_index} — Sample {index}\n"
        f"Color Perfect: {pixel_color_perfect:.4f} | Shape Perfect: {pixel_shape_perfect:.4f}"
    )
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    log("[DEBUG] Plot salvo como a.png")

    return pixel_color_perfect, pixel_shape_perfect



def plot_all_channels(output, title_prefix="Canal", save_prefix="debug_output"):
    """
    Plota todos os canais de uma matriz 3D no formato (30, 30, canais)
    """
    import matplotlib.pyplot as plt
    canais = output.shape[-1]
    for i in range(canais):
        canal = output[:, :, i]
        plt.figure(figsize=(4, 4))
        plt.imshow(canal, cmap="viridis", vmin=0, vmax=9, interpolation="nearest")
        plt.title(f"{title_prefix} {i}")
        plt.axis("off")
        filename = f"{save_prefix}_canal_{i}.png"
        plt.savefig(filename)
        plt.close()
        log(f"[DEBUG] Canal {i} salvo como {filename}")






def gerar_video_time_lapse(pasta="votos_visuais", block_idx=0, saida="video_votos.avi", fps=1):
    arquivos = sorted(glob.glob(os.path.join(pasta, f"{block_idx} - votos_iter_*.png")))

    if not arquivos:
        log("[VISUAL] Nenhuma imagem de iteração encontrada para gerar o vídeo.")
        return

    primeira_img = cv2.imread(arquivos[0])
    altura, largura, _ = primeira_img.shape
    video = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*'XVID'), fps, (largura, altura))

    for arq in arquivos:
        img = cv2.imread(arq)
        video.write(img)

    video.release()
    log(f"[VISUAL] Vídeo salvo em {saida}")
    return saida

def embutir_trilha_sonora(video_path="court_drama.mp4", block_idx=0, musica_path="intergalactic.mp3", output="court_with_sound.mp4"):
    try:
        if isinstance(video_path, int):
            video_path = f"videos/julgamento_block_{video_path}.mp4"

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")

        video = mpy.VideoFileClip(video_path)
        audio = mpy.AudioFileClip(musica_path).set_duration(video.duration)
        final = video.set_audio(audio)

        filename = f"{block_idx}_{output}" if isinstance(block_idx, int) else output
        final.write_videofile(filename, codec='libx264', audio_codec='aac')
        log(f"[AUDIO] Vídeo com trilha salvo em: {filename}")

    except Exception as e:
        log(f"[ERROR] Falha ao embutir trilha sonora: {e}")
