# metrics_utils.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from runtime_utils import log
from scipy.stats import entropy
import cv2
import glob
import sys
sys.path.append('/home/vscode/.local/lib/python3.10/site-packages')
import moviepy.editor as mpy

os.makedirs("images", exist_ok=True)
sns.set(style="whitegrid", font_scale=1.2)

def ensure_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else x

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
        if isinstance(v, tf.Tensor):
            v = v.numpy()
        if v.ndim == 5:
            v = v[0]
        if v.ndim == 4:
            v = extrair_matriz_simbolica(v)
        elif v.ndim == 3 and v.shape[-1] == 1:
            v = extrair_matriz_simbolica(v)
        elif v.ndim == 3:
            v = v[:, :, 0]
        return np.squeeze(v)
    except Exception as e:
        log(f"[VISUAL] Erro ao extrair matriz simbólica: {e}")
        return np.zeros((30, 30))


def salvar_voto_visual(votos, iteracao, block_idx, input_tensor_outros, idx=0, task_id=None, saida_dir="debug_plots"):
    import math
    os.makedirs(saida_dir, exist_ok=True)

    prefixo = f"{task_id}_" if task_id else ""
    fname = f"a.png"
    filepath = os.path.join(saida_dir, fname)

    votos_classes = []
    for i, v in enumerate(votos):
        if v is None:
            log(f"[VISUAL] ⚠️ Voto modelo_{i} é None. Ignorado.")
            continue
        try:
            if isinstance(v, tf.Tensor):
                v = v.numpy()

            if v.ndim == 5:
                v = v[0]  # remove batch

            if v.ndim == 3 and v.shape[-1] == 10:
                v_cls = np.argmax(v, axis=-1).astype(np.int32)
            elif v.ndim == 3:
                v_cls = extrair_matriz_simbolica(v)
            elif v.ndim == 2:
                v_cls = v.astype(np.int32)
            else:
                raise ValueError(f"[VISUAL] Formato de voto não suportado: {v.shape}")

            if i == 6:
                v_cls = 9 - v_cls

            log(f"[VISUAL DEBUG] modelo_{i}: únicos = {np.unique(v_cls)} shape={v_cls.shape}")
            votos_classes.append(v_cls)
        except Exception as e:
            log(f"[VISUAL] Erro ao preparar voto do modelo_{i}: {e}")

    if not votos_classes:
        log("[VISUAL] ❌ Nenhuma predição válida para visualização.")
        return

    try:
        input_vis = input_tensor_outros
        if isinstance(input_vis, tf.Tensor):
            input_vis = input_vis.numpy()

        if input_vis.ndim == 5:
            input_vis = input_vis[0]

        if input_vis.ndim == 3 and input_vis.shape[-1] == 10:
            input_vis = np.argmax(input_vis, axis=-1).astype(np.int32)
        elif input_vis.ndim == 3:
            input_vis = extrair_matriz_simbolica(input_vis)
        elif input_vis.ndim == 2:
            input_vis = input_vis.astype(np.int32)
        else:
            input_vis = np.zeros((30, 30))

        if input_vis.size != 900:
            log(f"[VISUAL] ⚠️ input_vis com {input_vis.size} elementos. Substituindo por zeros.")
            input_vis = np.zeros((30, 30))

    except Exception as e:
        log(f"[VISUAL] Erro ao preparar input_vis: {e}")
        input_vis = np.zeros((30, 30))

    votos_stack = np.stack(votos_classes, axis=0)
    h, w = votos_stack.shape[1:]
    entropia_map = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            _, counts = np.unique(votos_stack[:, i, j], return_counts=True)
            probs = counts / counts.sum()
            entropia_map[i, j] = entropy(probs, base=2)

    num_modelos = len(votos_classes)
    total_plots = num_modelos + 2
    cols = 4
    rows = math.ceil(total_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    cargos = {
        0: "Jurada 1", 1: "Jurada 2", 2: "Jurada 3",
        3: "Advogada", 4: "Juíza", 5: "Suprema Juíza", 6: "Promotor"
    }
    nomes = [cargos.get(i, f"Modelo {i}") for i in range(num_modelos)]

    for ax, voto, nome in zip(axes[:num_modelos], votos_classes, nomes):
        ax.imshow(voto, cmap="viridis", vmin=0, vmax=9, interpolation="nearest")
        ax.set_title(f"{nome}", fontsize=10)
        ax.axis("off")

    axes[num_modelos].imshow(input_vis, cmap="viridis", vmin=0, vmax=9, interpolation="nearest")
    axes[num_modelos].set_title("Input", fontsize=10)
    axes[num_modelos].axis("off")

    im = axes[num_modelos + 1].imshow(entropia_map, cmap="inferno", interpolation="nearest", vmin=0, vmax=np.log2(num_modelos))
    axes[num_modelos + 1].set_title("Entropia por Pixel", fontsize=10)
    axes[num_modelos + 1].axis("off")
    fig.colorbar(im, ax=axes[num_modelos + 1], fraction=0.046, pad=0.04)

    for i in range(num_modelos + 2, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Task {task_id} — Modelos — Iteração {iteracao} — Bloco {block_idx}", fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    log(f"[VISUAL] ✅ Mapa de votos + entropia salvo em {filepath}")



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


def extrair_matriz_simbolica(grid_3d, pad_value=0):
    return grid_3d[:, :, 0].astype(np.int32)




def plot_prediction_debug(
    raw_input,
    expected_output,
    predicted_output,
    model_index="",
    pad_value=0,
    index=0,
    task_id=None,
    output_dir="debug_plots"
):
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs(output_dir, exist_ok=True)

        expected_output = expected_output.numpy() if hasattr(expected_output, 'numpy') else expected_output
        predicted_output = predicted_output.numpy() if hasattr(predicted_output, 'numpy') else predicted_output

        if expected_output.ndim == 5:
            expected_output = expected_output[0]
        if predicted_output.ndim == 5:
            predicted_output = predicted_output[0]

        if expected_output.ndim == 3 and expected_output.shape[-1] == 10:
            # log("[DEBUG] Grid expected completa:")
            # log(f"[expected_output[:,:,0]]\n{expected_output[:,:,0]}")
            expected_output = extrair_matriz_simbolica(expected_output, -1)

        if predicted_output.ndim == 3:
            if predicted_output.shape[-1] == 10:
                # log("[DEBUG] Grid predicted completa:")
                # log(f"[predicted_output[:,:,0]]\n{predicted_output[:,:,0]}")
                predicted_output = predicted_output[:, :, 0]
            elif predicted_output.shape[-1] == 1:
                predicted_output = extrair_matriz_simbolica(predicted_output, -1)

        # log(f"[DEBUG] EXPECTED SHAPE FINAL: {expected_output.shape}")
        # log(f"[DEBUG] PREDICTED SHAPE FINAL: {predicted_output.shape}")

        if expected_output.ndim != 2 or predicted_output.ndim != 2:
            log(f"[ERROR] Shape incompatível para plot: predicted {predicted_output.shape}, expected {expected_output.shape}")
            return None

        # log(f"[DEBUG] Valores únicos esperados (final): {np.unique(expected_output)}")
        # log(f"[DEBUG] Valores únicos previstos (final): {np.unique(predicted_output)}")

        pixel_color_perfect, pixel_shape_perfect = compute_pixelwise_accuracy(
            expected_output, predicted_output, pad_value=pad_value
        )

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(expected_output, cmap="viridis", vmin=0, vmax=9, interpolation="nearest")
        ax[0].set_title("Expected Output")
        ax[0].axis("off")

        ax[1].imshow(predicted_output, cmap="viridis", vmin=0, vmax=9, interpolation="nearest")
        ax[1].set_title("Predicted Output")
        ax[1].axis("off")

        plt.suptitle(f"Task {task_id} — Model {model_index} — Sample {index}\n Color Perfect: {pixel_color_perfect:.4f} | Shape Perfect: {pixel_shape_perfect:.4f}")
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"a.png")
        # output_path = os.path.join(output_dir, f"task_{task_id}_model_{model_index}_sample_{index}.png")
        plt.savefig(output_path)
        plt.close()

        log(f"[DEBUG] Plot de debug salvo em: {output_path}")
        return pixel_color_perfect, pixel_shape_perfect

    except Exception as e:
        log(f"[ERROR] Falha ao gerar plot de debug: {e}")
        return None




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
