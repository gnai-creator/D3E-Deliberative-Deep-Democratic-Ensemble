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

import tensorflow as tf

class MaskedIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, ignore_class=None, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.intersections = self.add_weight(
            name="intersections",
            shape=(num_classes,),
            initializer="zeros",
            dtype=tf.float32
        )
        self.unions = self.add_weight(
            name="unions",
            shape=(num_classes,),
            initializer="zeros",
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1) if y_pred.shape.ndims == y_true.shape.ndims + 1 else y_pred
        y_pred = tf.cast(y_pred, tf.int32)

        for class_id in tf.range(self.num_classes, dtype=tf.int32):
            def update_class():
                true_mask = tf.equal(y_true, class_id)
                pred_mask = tf.equal(y_pred, class_id)
                intersection = tf.reduce_sum(tf.cast(true_mask & pred_mask, tf.float32))
                union = tf.reduce_sum(tf.cast(true_mask | pred_mask, tf.float32))

                updated_intersections = tf.tensor_scatter_nd_add(
                    self.intersections, indices=[[class_id]], updates=[intersection])
                updated_unions = tf.tensor_scatter_nd_add(
                    self.unions, indices=[[class_id]], updates=[union])

                self.intersections.assign(updated_intersections)
                self.unions.assign(updated_unions)

            tf.cond(class_id != self.ignore_class, update_class, lambda: None)

    def result(self):
        iou = tf.math.divide_no_nan(self.intersections, self.unions)
        if self.ignore_class is not None:
            mask = tf.one_hot(self.ignore_class, self.num_classes, on_value=False, off_value=True, dtype=tf.bool)
            iou = tf.boolean_mask(iou, mask)
        return tf.reduce_mean(iou)

    def reset_states(self):
        self.intersections.assign(tf.zeros_like(self.intersections))
        self.unions.assign(tf.zeros_like(self.unions))



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
    """
    Visualiza mapa de atenção médio a partir dos scores da camada MultiHeadAttention.
    Espera tensor de shape [1, heads, query_len, key_len], geralmente [1, 8, 900, 900].
    """
    if not isinstance(attn_tensor, tf.Tensor):
        log("[WARN] Atenção não é Tensor")
        return

    try:
        if len(attn_tensor.shape) == 4:
            # Média sobre os heads: [1, query_len, key_len]
            mean_heads = tf.reduce_mean(attn_tensor, axis=1)
            # Média sobre as posições de query: [1, key_len]
            mean_attention = tf.reduce_mean(mean_heads, axis=1)
            # Reshape para 30x30 se possível
            spatial_size = int(np.sqrt(mean_attention.shape[-1]))
            attn_map = tf.reshape(mean_attention[0], (spatial_size, spatial_size)).numpy()
        else:
            log(f"[WARN] Attention tensor com shape inesperado: {attn_tensor.shape}")
            return

        plt.figure(figsize=(6, 6))
        plt.imshow(attn_map, cmap='magma')
        plt.title(title)
        plt.axis('off')
        filename = f"images/attention_map_model_{model_index}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        log(f"[INFO] Attention map salvo: {filename}")

    except Exception as e:
        log(f"[ERRO] Falha ao visualizar attention map: {e}")


def plot_logit_distribution(logits, model_index="model"):
    # logits: Tensor com shape [1, H, W, num_classes]
    logits_np = logits.numpy().reshape(-1, logits.shape[-1])  # [H*W, num_classes]
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=logits_np, inner='point')
    plt.title(f"Distribuição dos logits por classe - {model_index}")
    plt.xlabel("Classe")
    plt.ylabel("Valor do Logit")
    plt.tight_layout()
    filename = f"images/logit_distribution_{model_index}.png"
    plt.savefig(filename)
    plt.close()
    print(f"[INFO] Logit distribution salva em {filename}")