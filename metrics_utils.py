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
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    import json

    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()

    # Ignora classe -1 (padding) e 0 (o carpete inútil)
    mask = (y_true_flat != -1) & (y_true_flat != 0)
    y_true_filtered = y_true_flat[mask]
    y_pred_filtered = y_pred_flat[mask]

    # Classes restantes depois da exclusão de 0
    remaining_classes = sorted(set(y_true_filtered) | set(y_pred_filtered))

    if len(remaining_classes) == 0:
        log("[WARN] Nenhuma classe relevante encontrada após filtragem. Nada a plotar.")
        return

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=remaining_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=remaining_classes, yticklabels=remaining_classes
    )
    plt.title("SageAxiom Confusion Matrix (sem classe 0)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    filename = f"images/confusion_matrix_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    log(f"[INFO] Matriz de confusão salva: {filename}")

    # Relatório de métricas por classe (sem 0)
    report = classification_report(
        y_true_filtered, y_pred_filtered,
        labels=remaining_classes,
        output_dict=True,
        zero_division=0
    )

    with open("images/per_class_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    log("[INFO] Relatório de métricas por classe salvo (sem classe 0): images/per_class_metrics.json")


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
    # Se input for one-hot (3D com último dim == VOCAB_SIZE), faz argmax
    if len(input_tensor.shape) == 3 and input_tensor.shape[-1] == 10:
        input_tensor = tf.argmax(input_tensor, axis=-1)

    input_img = input_tensor.numpy()
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