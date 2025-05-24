# metrics_utils.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from runtime_utils import log, make_serializable

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


def plot_prediction_debug(expected_output, predicted_output, raw_input=None, model_index="output", task_id="", index=0, pad_value=0):
    try:
        predicted_output = ensure_numpy(predicted_output).astype(np.int32)
        expected_output = ensure_numpy(expected_output).astype(np.int32)

        if raw_input is not None:
            input_img = np.asarray(raw_input)

        h = max(input_img.shape[0], predicted_output.shape[0], expected_output.shape[0])
        w = max(input_img.shape[1], predicted_output.shape[1], expected_output.shape[1])
        def pad(x): return np.pad(x, ((0, h - x.shape[0]), (0, w - x.shape[1])), constant_values=pad_value)

        input_img = pad(input_img)
        predicted_output = pad(predicted_output)
        expected_output = pad(expected_output)

        valid_mask = expected_output != pad_value
        pixel_color_perfect = np.mean((predicted_output == expected_output)[valid_mask])
        pixel_shape_perfect = np.mean((predicted_output > 0) == (expected_output > 0))
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


def plot_prediction_test(predicted_output, task_id, filename="output", raw_input=None, index=0, pad_value=0):
    try:
        predicted_output = ensure_numpy(predicted_output).astype(np.int32)
        if raw_input is not None:
            input_img = np.asarray(raw_input)

        h = max(input_img.shape[0], predicted_output.shape[0])
        w = max(input_img.shape[1], predicted_output.shape[1])
        def pad(x): return np.pad(x, ((0, h - x.shape[0]), (0, w - x.shape[1])), constant_values=pad_value)

        input_img = pad(input_img)
        predicted_output = pad(predicted_output)

        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        for ax, img, title in zip(axs, [input_img, predicted_output], ["Input", "Prediction"]):
            ax.imshow(img, cmap="viridis", vmin=0, vmax=9)
            ax.set_title(title)
            ax.axis("off")

        plt.suptitle(f"Prediction TEST - Model Task {index} ID: {task_id}", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs("images/test", exist_ok=True)
        full_filename = f"images/test/{filename}_Prediction TEST - Model Task {task_id}.png"
        plt.savefig(full_filename, dpi=150)
        plt.close()

        log(f"[INFO] Debug visual salvo: {full_filename}")
    except Exception as e:
        log(f"[ERROR] Falha ao gerar plot de debug: {e}")
