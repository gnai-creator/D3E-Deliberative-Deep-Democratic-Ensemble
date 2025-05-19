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

from tensorflow.keras.metrics import MeanIoU

class CustomMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes=15, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.iou = MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        self.iou.update_state(y_true, y_pred)

    def result(self):
        return self.iou.result()

    def reset_states(self):
        self.iou.reset_states()



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
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(10)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title("SageAxiom Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    filename = f"images/confusion_matrix_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    log(f"[INFO] Matriz de confusão salva: {filename}")

    report = classification_report(y_true_flat, y_pred_flat, labels=list(range(10)), output_dict=True)
    with open("images/per_class_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    log("[INFO] Relatório de métricas por classe salvo: images/per_class_metrics.json")

def plot_attempts_stats(task_times, attempts_per_task):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    tasks = list(task_times.keys())
    times = [task_times[t] for t in tasks]
    attempts = [attempts_per_task[t] for t in tasks]

    color = 'tab:blue'
    ax1.set_xlabel('Task ID')
    ax1.set_ylabel('Tempo (s)', color=color)
    ax1.bar(tasks, times, color=color, alpha=0.6, label="Tempo")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Tentativas', color=color)
    ax2.plot(tasks, attempts, color=color, marker='o', label="Tentativas")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Tempo e Tentativas por Task")
    plt.xticks(rotation=45)
    plt.savefig("images/task_performance_overview.png")
    plt.close()
    log("[INFO] Gráfico de performance salvo: images/task_performance_overview.png")

def plot_prediction_debug(input_tensor, expected_output, predicted_output, model_index):
    if input_tensor.shape[-1] > 10:
        input_tensor = tf.one_hot(tf.argmax(input_tensor, axis=-1), depth=10)
    input_img = input_tensor.numpy().argmax(axis=-1)
    expected_img = expected_output.numpy()
    prediction_img = predicted_output
    heatmap = (prediction_img == expected_img).astype(int)

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(input_img, cmap='viridis')
    axs[0].set_title("Input")
    axs[1].imshow(expected_img, cmap='viridis')
    axs[1].set_title("Expected Output")
    axs[2].imshow(prediction_img, cmap='viridis')
    axs[2].set_title("Prediction")
    axs[3].imshow(heatmap, cmap='gray')
    axs[3].set_title("Correctness Heatmap")
    for ax in axs:
        ax.axis('off')

    plt.suptitle(f"Prediction Debug - Model {model_index}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"images/prediction_debug_{model_index}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    log(f"[INFO] Debug visual salvo: {filename}")

def visualize_attention_map(attn_tensor, model_index, title="Attention Output"):
    if isinstance(attn_tensor, tf.Tensor):
        attn_map = tf.reduce_mean(attn_tensor, axis=-1).numpy()[0]
    else:
        log("[WARN] Atenção não é Tensor")
        return
    plt.figure(figsize=(6, 6))
    plt.imshow(attn_map, cmap='magma')
    plt.title(title)
    plt.axis('off')
    filename = f"images/attention_map_model_{model_index}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    log(f"[INFO] Attention map salvo: {filename}")