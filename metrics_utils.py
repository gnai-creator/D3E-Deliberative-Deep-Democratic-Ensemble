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

def safe_squeeze_axis(tensor, axis):
    if tensor.shape[axis] == 1:
        return tf.squeeze(tensor, axis=axis)
    return tensor

def preparar_voto_para_visualizacao(v):
    v = ensure_numpy(v)

    # Se ainda tiver logits (mais de 2 dimensões), transforma em classes
    if v.ndim > 2:
        v = np.argmax(v, axis=-1)

    # Remove eixos extras até ficar 2D
    while v.ndim > 2:
        v = np.squeeze(v, axis=0)

    if v.ndim == 3:
        v = v[..., 0]

    if v.size != 900:
        log(f"[VISUAL] ⚠️ Voto inválido com {v.size} elementos. Esperado 900 para shape (30, 30).")
        return np.zeros((30, 30), dtype=np.int32)

    return v.reshape((30, 30)).astype(np.int32)


def salvar_voto_visual(votos, iteracao, block_idx, input_tensor_outros, idx=0, task_id=None, saida_dir="votos_visuais"):
    os.makedirs(saida_dir, exist_ok=True)
    prefixo = f"{task_id}_" if task_id else ""
    fname = f"{prefixo}{block_idx} - votos_iter_{iteracao:02d}.png"
    filepath = os.path.join(saida_dir, fname)

    votos_classes = []
    for i, v in enumerate(votos):
        if v is None:
            continue
        try:
            v_cls = preparar_voto_para_visualizacao(v)
            log(f"[VISUAL DEBUG] modelo_{i}: únicos = {np.unique(v_cls)}")
            votos_classes.append(v_cls)
        except Exception as e:
            log(f"[VISUAL] Erro ao preparar voto do modelo_{i}: {e}")

    num_modelos = len(votos_classes)

    if not votos_classes:
        log("[VISUAL] Nenhuma predição válida para visualização.")
        fig, ax = plt.subplots(figsize=(6, 6))

        raw_input = np.squeeze(input_tensor_outros[0])
        if raw_input.ndim == 3:
            H, W, C = raw_input.shape
            heatmap = np.zeros((H, W))
            for i in range(H):
                for j in range(W):
                    vals, counts = np.unique(raw_input[i, j, :], return_counts=True)
                    probs = counts / np.sum(counts)
                    entropy = -np.sum(probs * np.log2(probs + 1e-9))
                    heatmap[i, j] = entropy
            sns.heatmap(heatmap, ax=ax, cmap="magma", square=True, cbar=True)
            ax.set_title("Entropia dos Pixels (Sem votos válidos)", fontsize=10)
        else:
            ax.text(0.5, 0.5, "Sem votos válidos", ha="center", va="center", fontsize=14)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        log(f"[VISUAL] Salvou fallback visual em {filepath}")
        return

    # Mapa de consenso
    votos_stack = np.stack(votos_classes, axis=0)
    consenso_map = np.zeros((30, 30), dtype=np.uint8)
    for i in range(30):
        for j in range(30):
            _, counts = np.unique(votos_stack[:, i, j], return_counts=True)
            if np.max(counts) >= 3:
                consenso_map[i, j] = 1

    fig, axes = plt.subplots(1, num_modelos + 2, figsize=(4 * (num_modelos + 2), 4))

    cargos = {
        0: "Jurada 1", 1: "Jurada 2", 2: "Jurada 3",
        3: "Advogada", 4: "Juíza", 5: "Suprema Juíza"
    }
    nomes = [cargos.get(i, f"Modelo {i}") for i in range(num_modelos)]

    for ax, voto, nome in zip(axes[:num_modelos], votos_classes, nomes):
        sns.heatmap(voto, ax=ax, cbar=False, cmap="viridis", square=True, vmin=0, vmax=9)
        ax.set_title(f"{nome}", fontsize=10)
        ax.axis("off")

    # Prepara input visual
    try:
        input_vis = ensure_numpy(input_tensor_outros)
        if input_vis.ndim == 5:
            input_vis = tf.argmax(input_vis[0], axis=-1).numpy()

        elif input_vis.ndim == 4:
            input_vis = input_vis[0, :, :, 0]
        elif input_vis.ndim == 3 and input_vis.shape[-1] > 1:
            input_vis = input_vis[..., 0]
        input_vis = np.squeeze(input_vis)

        if input_vis.size != 900:
            log(f"[VISUAL] ⚠️ input_vis com {input_vis.size} elementos. Substituindo por zeros.")
            input_vis = np.zeros((30, 30))
        else:
            input_vis = input_vis.reshape((30, 30))

    except Exception as e:
        log(f"[VISUAL] Erro ao preparar input_vis: {e}")
        input_vis = np.zeros((30, 30))

    sns.heatmap(input_vis, ax=axes[-2], cbar=False, cmap="viridis", square=True, vmin=0, vmax=9)
    axes[-2].set_title("Input", fontsize=10)
    axes[-2].axis("off")

    sns.heatmap(consenso_map, ax=axes[-1], cbar=False, cmap="Greens", square=True)
    axes[-1].set_title("Mapa de Consenso (≥3)", fontsize=10)
    axes[-1].axis("off")

    plt.suptitle(f"Predições dos Modelos - Iteração {iteracao}", fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    log(f"[VISUAL] Salvo mapa de votos + consenso em {filepath}")





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
