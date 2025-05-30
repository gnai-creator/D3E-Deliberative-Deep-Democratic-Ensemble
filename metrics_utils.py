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

def extrair_matriz_simbolica(grid_3d, pad_value=-1):
    return grid_3d[:, :, 0].astype(np.int32)

def extrair_matriz_simbolica_test(grid_3d, pad_value=-1):
    if grid_3d.shape[-1] == 1:
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
    # log(f"[DEBUG] prepare_display_image - input shape: {img.shape}")
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
        log(f"[DEBUG] prepare_display_image - unhandled shape: {img.shape}")
        return np.full((h, w, 3), pad_value)

def plot_prediction_test(predicted_output, raw_input, pad_value, save_path):
    try:
        log(f"[DEBUG] plot_prediction_test - predicted_output original shape: {getattr(predicted_output, 'shape', 'N/A')}")
        predicted_output = prepare_display_image(predicted_output, pad_value, 30, 30)
        log(f"[DEBUG] plot_prediction_test - predicted_output processed shape: {predicted_output.shape}")

        if raw_input is not None:
            log(f"[DEBUG] plot_prediction_test - raw_input original shape: {getattr(raw_input, 'shape', 'N/A')}")
            raw_input = prepare_display_image(raw_input, pad_value, 30, 30)
            log(f"[DEBUG] plot_prediction_test - raw_input processed shape: {raw_input.shape}")
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
        log(f"[INFO] Resultado do teste salvo em: {save_path}")
        plt.close()

    except Exception as e:
        log("[ERROR] Falha ao gerar plot de teste:", str(e))



def ensure_numpy(tensor):
    return tensor.numpy() if hasattr(tensor, "numpy") else tensor

def preparar_voto_para_visualizacao(voto):
    try:
        voto = tf.convert_to_tensor(voto)
        log(f"[DEBUG] PREPARAR VOTO PARA VIZUALIZAÇÃO VOTO SHAPE {voto.shape}")

        # Caso (1, 30, 30, 1, 10): aplicar argmax no último eixo
        if voto.shape.rank == 5 and voto.shape[-1] == 10:
            voto = tf.argmax(voto, axis=-1)  # (1, 30, 30, 1)

        # Caso (1, 30, 30, 1, 1): remover última dimensão
        elif voto.shape.rank == 5 and voto.shape[-1] == 1:
            voto = tf.squeeze(voto, axis=-1)  # (1, 30, 30, 1)

        # Se ainda estiver em shape inesperado, tentar normalizar
        if voto.shape.rank == 3:
            voto = tf.expand_dims(voto, axis=-1)  # (1, 30, 30, 1)
        elif voto.shape.rank == 2:
            voto = tf.reshape(voto, (1, 30, 30, 1))

        # Confirma formato final
        if voto.shape != (1, 30, 30, 3):
            log(f"[VISUAL] ⚠️ Voto visual com shape inesperado: {voto.shape}")
            return None

        return voto

    except Exception as e:
        log(f"[VISUAL] Erro ao preparar voto: {e}")
        return None





def ensure_numpy(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def compute_pixelwise_accuracy(expected, predicted, pad_value=-1):
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
    pad_value=-1,
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
    if predicted_output.ndim == 3:
        predicted_2d = predicted_output[:, :, 0]
    elif predicted_output.ndim == 2:
        predicted_2d = predicted_output
    else:
        raise ValueError(f"[ERRO] predicted_output com shape inesperado: {predicted_output.shape}")


    if expected_2d.shape != predicted_2d.shape:
        log(f"[ERROR] Shape incompatível: predicted={predicted_2d.shape}, expected={expected_2d.shape}")
        return 0, 0

    # Métricas
    mask = expected_2d != pad_value
    total = np.sum(mask)
    correct = np.sum((expected_2d == predicted_2d) & mask)
    pixel_color_perfect = correct / total if total > 0 else 0.0
    pixel_shape_perfect = np.sum((expected_2d == predicted_2d) & mask) / np.sum(mask) if np.sum(mask) > 0 else 0.0

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
