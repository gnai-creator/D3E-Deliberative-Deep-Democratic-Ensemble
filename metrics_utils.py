# metrics_utils.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from runtime_utils import log

sns.set(style="whitegrid", font_scale=1.2)


def plot_history(history, model_name):
    plt.figure(figsize=(12, 6))
    for key in history.history:
        plt.plot(history.history[key], label=key)
    plt.title("\nSageAxiom Training History", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss / Metric", fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = f"training_plot_{model_name}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    log(f"[INFO] Plot do treinamento salvo: {filename}")


def plot_confusion(y_true, y_pred, model_name):
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=list(range(10)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, square=True,
                xticklabels=range(10), yticklabels=range(10), cbar_kws={"shrink": 0.8})
    plt.title("SageAxiom Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.tight_layout()
    filename = f"confusion_matrix_{model_name}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    log(f"[INFO] Matriz de confusão salva: {filename}")

    report = classification_report(y_true_flat, y_pred_flat, labels=list(range(10)), output_dict=True)
    with open("per_class_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    log("[INFO] Relatório de métricas por classe salvo: per_class_metrics.json")


def plot_attempts_stats(task_times, attempts_per_task):
    fig, ax1 = plt.subplots(figsize=(14, 6))
    tasks = list(task_times.keys())
    times = [task_times[t] for t in tasks]
    attempts = [attempts_per_task[t] for t in tasks]

    color = 'tab:blue'
    ax1.set_xlabel('Task ID', fontsize=12)
    ax1.set_ylabel('Tempo (s)', color=color, fontsize=12)
    ax1.bar(tasks, times, color=color, alpha=0.6, label="Tempo")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Tentativas', color=color, fontsize=12)
    ax2.plot(tasks, attempts, color=color, marker='o', label="Tentativas")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Tempo e Tentativas por Task", fontsize=16)
    plt.savefig("task_performance_overview.png", dpi=150)
    plt.close()
    log("[INFO] Gráfico de performance salvo: task_performance_overview.png")


def plot_prediction_debug(input_tensor, expected_output, predicted_output, model_index):
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
    filename = f"prediction_debug_{model_index}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    log(f"[INFO] Debug visual salvo: {filename}")