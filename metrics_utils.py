# metrics_utils.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from runtime_utils import log

# Cria diretório para imagens
os.makedirs("images", exist_ok=True)
sns.set(style="whitegrid", font_scale=1.2)



def plot_training_input(input_tensor, model_name):
    """
    Visualiza a primeira amostra de input em [B, H, W, C] como imagem com argmax no canal.
    Se for [H, W, C], usa direto. Se for [B, H, W, C], usa o primeiro item do batch.
    """
    input_tensor = tf.convert_to_tensor(input_tensor)

    if len(input_tensor.shape) == 4:
        # Assume [B, H, W, C]
        input_tensor = input_tensor[0]
    elif len(input_tensor.shape) != 3:
        raise ValueError(
            f"[ERROR] input_tensor deve ter shape [H, W, C] ou [B, H, W, C], mas tem {input_tensor.shape}"
        )

    # input_tensor: [H, W, C] com one-hot codificado
    input_visual = tf.argmax(input_tensor, axis=-1).numpy().astype(np.int32)  # [H, W]

    plt.figure(figsize=(4, 4))
    plt.imshow(input_visual, cmap='viridis')
    plt.title("Input Visualizado - argmax por canal")
    plt.colorbar()

    filename = f"images/input_visual_{model_name}.png"
    plt.savefig(filename)
    plt.close()

    log(f"[INFO] Input visualizado salvo: {filename}")





def plot_prediction_debug(input_tensor, expected_output, predicted_output, model_index, index, pad_value=0):
    def to_numpy_safe(x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        return np.asarray(x)

    try:
        os.makedirs("images", exist_ok=True)

        input_tensor = to_numpy_safe(input_tensor)
        expected_output = to_numpy_safe(expected_output).astype(np.int32)
        predicted_output = to_numpy_safe(predicted_output).astype(np.int32)

        if input_tensor.ndim == 4 and input_tensor.shape[2] == 1:
            input_tensor = np.squeeze(input_tensor, axis=2)  # (H, W, C)
        if input_tensor.ndim == 3 and input_tensor.shape[-1] == 10:
            input_img = np.argmax(input_tensor, axis=-1)
        else:
            raise ValueError(f"[ERROR] input_tensor shape inesperado: {input_tensor.shape}")

        if predicted_output.ndim == 4:
            predicted_output = predicted_output[0]
        if predicted_output.ndim == 3 and predicted_output.shape[-1] == 1:
            predicted_output = predicted_output[:, :, 0]

        if expected_output.ndim == 3 and expected_output.shape[-1] == 1:
            expected_output = expected_output[:, :, 0]

        if not all(img.ndim == 2 for img in [input_img, expected_output, predicted_output]):
            raise ValueError("[ERROR] Todos os dados esperados no formato 2D (H, W).")

        valid_mask = expected_output != pad_value
        pixel_color_perfect = np.mean((predicted_output == expected_output)[valid_mask])
        pixel_shape_perfect = np.mean((predicted_output > 0) == (expected_output > 0))

        heatmap = ((predicted_output > 0) == (expected_output > 0)).astype(np.int32)

        
        sns.set(style="whitegrid")

        log(f"input_img.shape: {input_img.shape}")
        log(f"predicted_output.shape: {predicted_output.shape}")
        log(f"expected_output.shape: {expected_output.shape}")


        fig, axs = plt.subplots(1, 4, figsize=(22, 4))
        titles = [
            "Input",
            "Expected Output",
            f"Prediction\n(Color Match: {pixel_color_perfect:.2%})",
            f"Shape Match\n(Presence: {pixel_shape_perfect:.2%})"
        ]
        images = [input_img, expected_output, predicted_output, heatmap]
        cmaps = ["viridis", "viridis", "viridis", "gray"]

        for ax, img, title, cmap in zip(axs, images, titles, cmaps):
            ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
            ax.set_title(title)
            ax.axis("off")

        plt.suptitle(f"Prediction Debug - Model {model_index}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"images/prediction_debug_{model_index}.png"
        plt.savefig(filename, dpi=150)
        plt.close()

       
        log(f"[INFO] Debug visual salvo: {filename}")

        return pixel_color_perfect, pixel_shape_perfect

    except Exception as e:
        log(f"[ERROR] Falha ao gerar plot de debug: {e}")





def plot_raw_input_preview(tensor_raw, model_name):


    try:
        tensor_raw = tf.convert_to_tensor(tensor_raw)

        if tensor_raw.shape.rank not in (2, 3):
            log(f"[ERROR] Tensor com shape inválido para visualização: {tensor_raw.shape}")
            return

        # Se for 3D com canal maior que 1, converter usando argmax
        if tensor_raw.shape.rank == 3 and tensor_raw.shape[-1] > 1:
            tensor_raw = tf.argmax(tensor_raw, axis=-1)

        array = tensor_raw.numpy().astype(np.int32)

        plt.figure(figsize=(4, 4))
        plt.imshow(array, cmap="viridis", interpolation="nearest")
        plt.title("Input Original (com padding)")
        plt.colorbar()
        filename = f"images/input_preview_raw_{model_name}.png"
        plt.savefig(filename)
        plt.close()
        log(f"[INFO] Prévia do input bruto salva: {filename}")

    except Exception as e:
        log(f"[ERROR] Falha ao plotar input bruto: {e}")






def plot_history(history, model_name):
    plt.figure(figsize=(10, 5))
    for key in history.history:
        plt.plot(history.history[key], label=key)
    plt.title("SageAxiom Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"images/training_plot_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    log(f"[INFO] Plot do treinamento salvo: {filename}")

def plot_confusion(y_true, y_pred, model_name):


    # Sempre flattens seguros
    y_true_flat = np.array(y_true).reshape(-1)
    y_pred_flat = np.array(y_pred).reshape(-1)

    if y_true_flat.shape != y_pred_flat.shape:
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]

    # Ignora PAD (-1) e classe 0
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
    plt.title("SageAxiom Confusion Matrix (sem classe 0)")
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
        json.dump(report, f, indent=2)
    log("[INFO] Relatório de métricas por classe salvo (sem classe 0): images/per_class_metrics.json")



def ensure_numpy(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def plot_prediction_test(input_tensor, predicted_output, task_id, filename="output", index=0, pad_value=0):
    try:
        if isinstance(input_tensor, tf.Tensor):
            input_tensor = input_tensor.numpy()
        if isinstance(predicted_output, tf.Tensor):
            predicted_output = predicted_output.numpy()

        input_tensor = np.asarray(input_tensor)
        predicted_output = np.asarray(predicted_output).astype(np.int32)

        # Convert one-hot input to class indices if needed
        if input_tensor.ndim == 4 and input_tensor.shape[-1] == 10:
            input_img = np.argmax(input_tensor, axis=-1)[0]
        elif input_tensor.ndim == 3 and input_tensor.shape[-1] == 10:
            input_img = np.argmax(input_tensor, axis=-1)
        elif input_tensor.ndim == 2:
            input_img = input_tensor
        elif input_tensor.ndim == 3 and input_tensor.shape[-1] == 1:
            input_img = input_tensor[:, :, 0]
        else:
            input_img = input_tensor.copy()

        if predicted_output.ndim == 4:
            predicted_output = predicted_output[0]
        if predicted_output.ndim == 3 and predicted_output.shape[-1] == 1:
            predicted_output = predicted_output[:, :, 0]

        # Align shapes for plotting
        h = max(input_img.shape[0], predicted_output.shape[0])
        w = max(input_img.shape[1], predicted_output.shape[1])
        def pad(x, h, w):
            padded = np.full((h, w), pad_value, dtype=x.dtype)
            padded[:x.shape[0], :x.shape[1]] = x
            return padded

        input_img = pad(input_img, h, w)
        predicted_output = pad(predicted_output, h, w)

        sns.set(style="whitegrid")
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        titles = ["Input", "Prediction"]
        images = [input_img, predicted_output]
        cmaps = ["viridis", "viridis"]

        for ax, img, title, cmap in zip(axs, images, titles, cmaps):
            ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
            ax.set_title(title)
            ax.axis("off")

        file = f"Prediction TEST - Model Task {task_id}"
        plt.suptitle(file, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs("images/test", exist_ok=True)
        full_filename = f"images/test/{file}.png"
        plt.savefig(full_filename, dpi=150)
        plt.close()

        log(f"[INFO] Debug visual salvo: {full_filename}")

    except Exception as e:
        log(f"[ERROR] Falha ao gerar plot de debug: {e}")