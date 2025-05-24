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
# from moviepy.editor import VideoFileClip, AudioFileClip


os.makedirs("images", exist_ok=True)
sns.set(style="whitegrid", font_scale=1.2)


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

def prepare_display_image(img, pad_value, h, w):
    pad_height = h - img.shape[0]
    pad_width = w - img.shape[1]
    if img.ndim == 2:
        img = np.pad(img, ((0, pad_height), (0, pad_width)), constant_values=pad_value)
    elif img.ndim == 3:
        img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), constant_values=pad_value)
        if img.shape[-1] > 3:
            img = np.mean(img, axis=-1)  # Reduz todos os canais extras para média
        elif img.shape[-1] == 1:
            img = img[:, :, 0]  # Remove canal singleton
    else:
        raise ValueError(f"[pad()] Formato inesperado: {img.shape}")
    return img

def plot_prediction_debug(expected_output, predicted_output, raw_input=None, model_index="output", task_id="", index=0, pad_value=0):
    """
    Gera uma imagem com a comparação entre entrada, saída esperada e saída prevista,
    junto com uma máscara de acertos por presença (shape) e cor (classe).
    Retorna métricas de acurácia pixel a pixel.
    """
    try:
        # Converte tensores para numpy, se necessário
        predicted_output = ensure_numpy(predicted_output).astype(np.int32)
        expected_output = ensure_numpy(expected_output).astype(np.int32)
        input_img = np.asarray(raw_input) if raw_input is not None else np.zeros_like(expected_output)

        # Garante formato 2D removendo canais desnecessários
        if predicted_output.ndim == 3 and predicted_output.shape[-1] == 1:
            predicted_output = predicted_output[:, :, 0]
        if expected_output.ndim == 3 and expected_output.shape[-1] == 1:
            expected_output = expected_output[:, :, 0]

        # Considera todos os pixels como válidos (inclusive classe 0)
        valid_mask = np.ones_like(expected_output, dtype=bool)

        assert valid_mask.shape == predicted_output.shape, f"Máscara inválida: {valid_mask.shape} vs {predicted_output.shape}"

        # Cálculo das métricas principais (forçando casting para float para evitar truncamento inteiro)
        correct_pixels = (predicted_output == expected_output)[valid_mask]
        pixel_color_perfect = np.sum(correct_pixels) / correct_pixels.size

        correct_shape = ((predicted_output > 0) == (expected_output > 0))[valid_mask]
        pixel_shape_perfect = np.sum(correct_shape) / correct_shape.size

        # Calcula altura e largura máximas para padding
        h = max(input_img.shape[0], predicted_output.shape[0], expected_output.shape[0])
        w = max(input_img.shape[1], predicted_output.shape[1], expected_output.shape[1])

        # Padroniza os arrays para mesma dimensão para visualização
        input_img = prepare_display_image(input_img, pad_value, h, w)
        predicted_output = prepare_display_image(predicted_output, pad_value, h, w)
        expected_output = prepare_display_image(expected_output, pad_value, h, w)

        # Mapa binário de acertos de presença
        heatmap = ((predicted_output > 0) == (expected_output > 0)).astype(np.int32)

        # Construção visual do plot
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


def plot_prediction_test(predicted_output, task_id, filename="output", raw_input=None, index=0, pad_value=0):
    """
    Gera um gráfico de comparação entre entrada e predição final em fase de teste,
    útil para inspeção qualitativa dos resultados.
    """
    try:
        # Garante numpy e tipo inteiro para a previsão
        predicted_output = ensure_numpy(predicted_output).astype(np.int32)
        input_img = np.asarray(raw_input) if raw_input is not None else np.zeros_like(predicted_output)

        # Garante formato 2D removendo canais desnecessários
        if predicted_output.ndim == 3 and predicted_output.shape[-1] == 1:
            predicted_output = predicted_output[:, :, 0]
        if input_img.ndim == 3 and input_img.shape[-1] == 1:
            input_img = input_img[:, :, 0]

        h = max(input_img.shape[0], predicted_output.shape[0])
        w = max(input_img.shape[1], predicted_output.shape[1])

        input_img = prepare_display_image(input_img, pad_value, h, w)
        predicted_output = prepare_display_image(predicted_output, pad_value, h, w)

        # Cria subplots com input e saída prevista
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        for ax, img, title in zip(axs, [input_img, predicted_output], ["Input", "Prediction"]):
            ax.imshow(img, cmap="viridis", vmin=0, vmax=9)
            ax.set_title(title)
            ax.axis("off")

        plt.suptitle(f"Prediction TEST - Model Task {index} ID: {task_id}", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs("images/test", exist_ok=True)

        full_filename = f"images/test/{filename}_Prediction_TEST_Task_{task_id}.png"
        plt.savefig(full_filename, dpi=150)
        plt.close()
        log(f"[INFO] Debug visual salvo: {full_filename}")

    except Exception as e:
        log(f"[ERROR] Falha ao gerar plot de teste: {e}")





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

def gerar_video_time_lapse(pasta="votos_visuais", output="court_drama.mp4", fps=1):
    arquivos = sorted(glob.glob(f"{pasta}/votos_iter_*.png"))
    if not arquivos:
        log("[VISUAL] Nenhuma imagem de iteração encontrada para gerar o vídeo.")
        return

    log(f"[VIDEO] Gerando vídeo a partir de {len(arquivos)} quadros...")

    with imageio.get_writer(output, fps=fps) as writer:
        for img_path in arquivos:
            img = imageio.imread(img_path)
            writer.append_data(img)

    log(f"[VIDEO] Time-lapse salvo em: {output}")



def embutir_trilha_sonora(video_path="court_drama.mp4", musica_path="intergalactic.mp3", output_path="court_with_sound.mp4"):
    video = mpy.VideoFileClip(video_path)
    audio = mpy.AudioFileClip(musica_path).set_duration(video.duration)
    final = video.set_audio(audio)
    final.write_videofile(output_path, codec='libx264', audio_codec='aac')
    log(f"[AUDIO] Vídeo com trilha salvo em: {output_path}")