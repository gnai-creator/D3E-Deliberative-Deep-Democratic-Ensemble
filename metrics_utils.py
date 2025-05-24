# metrics_utils.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from runtime_utils import log, make_serializable
import imageio
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
    print(f"[DEBUG] prepare_display_image - input shape: {img.shape}")
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

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[INFO] Resultado do teste salvo em: {save_path}")
        plt.close()

    except Exception as e:
        print("[ERROR] Falha ao gerar plot de teste:", str(e))

def salvar_voto_visual(votos, iteracao, saida_dir="votos_visuais"):
    os.makedirs(saida_dir, exist_ok=True)
    num_modelos = len(votos)
    votos_classes = [np.argmax(ensure_numpy(v), axis=-1)[0] for v in votos if v is not None]

    if not votos_classes:
        print("[WARNING] Nenhuma predição válida para visualização.")
        return

    votos_stack = np.stack(votos_classes, axis=0)
    H, W = votos_stack.shape[1:3]
    consenso_map = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            _, counts = np.unique(votos_stack[:, i, j], return_counts=True)
            if np.max(counts) >= 3:
                consenso_map[i, j] = 1

    fig, axes = plt.subplots(1, num_modelos + 1, figsize=(4 * (num_modelos + 1), 4))

    nomes = [f"Jurada {i+1}" if i < 3 else "Advogada" if i == 3 else "Juíza" for i in range(num_modelos)]

    for ax, voto, nome in zip(axes[:-1], votos_classes, nomes):
        sns.heatmap(voto, ax=ax, cbar=False, cmap="viridis", square=True)
        ax.set_title(f"{nome}", fontsize=10)
        ax.axis("off")

    sns.heatmap(consenso_map, ax=axes[-1], cbar=False, cmap="Greens", square=True)
    axes[-1].set_title("Mapa de Consenso (≥3)", fontsize=10)
    axes[-1].axis("off")

    plt.suptitle(f"Predições dos Modelos - Iteração {iteracao}", fontsize=12)
    filepath = os.path.join(saida_dir, f"votos_iter_{iteracao:02d}.png")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"[VISUAL] Salvo mapa de votos + consenso em {filepath}")


def plot_confusion(y_true, y_pred, model_name):
    y_true_flat = np.array(y_true).reshape(-1)
    y_pred_flat = np.array(y_pred).reshape(-1)

    if y_true_flat.shape != y_pred_flat.shape:
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]

    mask = (y_true_flat != -1) & (y_true_flat != 0)
    y_true_filtered = y_true_flat[mask]
    y_pred_filtered = y_pred_flat[mask]

    remaining_classes = sorted(set(y_true_filtered) | set(y_pred_filtered))
    if len(remaining_classes) == 0:
        log("[WARN] Nenhuma classe relevante encontrada após filtragem. Nada a plotar.")
        return

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=remaining_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=remaining_classes, yticklabels=remaining_classes)
    plt.title("Confusion Matrix (sem classe 0)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    filename = f"images/confusion_matrix_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    log(f"[INFO] Matriz de confusão salva: {filename}")

    report = classification_report(
        y_true_filtered, y_pred_filtered,
        labels=remaining_classes, output_dict=True, zero_division=0
    )

    with open("images/per_class_metrics.json", "w") as f:
        json.dump(make_serializable(report), f, indent=2)

    log("[INFO] Relatório de métricas por classe salvo (sem classe 0): images/per_class_metrics.json")

def ensure_numpy(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def plot_prediction_debug(expected_output, predicted_output, raw_input=None, model_index="output", task_id="", index=0, pad_value=0):
    try:
        predicted_output = ensure_numpy(predicted_output).astype(np.int32)
        expected_output = ensure_numpy(expected_output).astype(np.int32)
        input_img = np.asarray(raw_input) if raw_input is not None else np.zeros_like(expected_output)

        if predicted_output.ndim == 3 and predicted_output.shape[-1] == 1:
            predicted_output = predicted_output[:, :, 0]
        if expected_output.ndim == 3 and expected_output.shape[-1] == 1:
            expected_output = expected_output[:, :, 0]

        valid_mask = np.ones_like(expected_output, dtype=bool)

        assert valid_mask.shape == predicted_output.shape, f"Máscara inválida: {valid_mask.shape} vs {predicted_output.shape}"

        correct_pixels = (predicted_output == expected_output)[valid_mask]
        pixel_color_perfect = np.sum(correct_pixels) / correct_pixels.size

        correct_shape = ((predicted_output > 0) == (expected_output > 0))[valid_mask]
        pixel_shape_perfect = np.sum(correct_shape) / correct_shape.size

        h = max(input_img.shape[0], predicted_output.shape[0], expected_output.shape[0])
        w = max(input_img.shape[1], predicted_output.shape[1], expected_output.shape[1])

        input_img = prepare_display_image(input_img, pad_value, h, w)
        predicted_output = prepare_display_image(predicted_output, pad_value, h, w)
        expected_output = prepare_display_image(expected_output, pad_value, h, w)

        heatmap = ((predicted_output > 0) == (expected_output > 0)).astype(np.int32)

        fig, axs = plt.subplots(1, 4, figsize=(22, 4))
        for ax, img, title, cmap in zip(
            axs,
            [input_img, expected_output, predicted_output, heatmap],
            ["Input", "Expected Output", f"Prediction\n(Color Match: {pixel_color_perfect:.2%})", f"Shape Match\n(Presence: {pixel_shape_perfect:.2%})"],
            ["viridis", "viridis", "viridis", "gray"]
        ):
            ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
            ax.set_title(title)
            ax.axis("off")

        plt.suptitle(f"Prediction Debug - Model {model_index} - {index} - Task: {task_id}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        filename = f"images/prediction_debug_{model_index}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        log(f"[INFO] Debug visual salvo: {filename}")

        return pixel_color_perfect, pixel_shape_perfect

    except Exception as e:
        log(f"[ERROR] Falha ao gerar plot de debug: {e}")


def gerar_video_time_lapse(pasta="votos_visuais", block_idx=0, output="court_drama.mp4", fps=1):
    try:
        if not isinstance(pasta, str):
            raise ValueError(f"[ERRO] Argumento 'pasta' deve ser uma string. Recebido: {pasta}")

        arquivos = sorted(glob.glob(os.path.join(pasta, "votos_iter_*.png")))
        if not arquivos:
            log("[VISUAL] Nenhuma imagem de iteração encontrada para gerar o vídeo.")
            return None

        os.makedirs("videos", exist_ok=True)
        filename = f"videos/julgamento_block_{block_idx}.mp4"

        log(f"[VIDEO] Gerando vídeo com {len(arquivos)} quadros...")
        with imageio.get_writer(filename, fps=fps) as writer:
            for img_path in arquivos:
                img = imageio.imread(img_path)
                writer.append_data(img)

        log(f"[VIDEO] Time-lapse salvo em: {filename}")
        return filename

    except Exception as e:
        log(f"[ERRO] Falha ao gerar vídeo: {e}")
        return None

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
