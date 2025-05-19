import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from runtime_utils import log

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
    filename = f"training_plot {model_name}.png"
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
    filename = f"confusion_matrix {model_name}.png"
    plt.savefig(filename)
    plt.close()
    log(f"[INFO] Matriz de confusão salva: {filename}")

    report = classification_report(y_true_flat, y_pred_flat, labels=list(range(10)), output_dict=True)
    with open("per_class_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    log("[INFO] Relatório de métricas por classe salvo: per_class_metrics.json")

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
    plt.savefig("task_performance_overview.png")
    plt.close()
    log("[INFO] Gráfico de performance salvo: task_performance_overview.png")

def plot_prediction_debug(input_tensor, expected_output, predicted_output, model_index):
    input_img = input_tensor.numpy().argmax(axis=-1)
    expected_img = expected_output.numpy()
    prediction_img = predicted_output
    heatmap = (prediction_img == expected_img).astype(int)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(input_img, cmap='viridis')
    axs[0].set_title("Input")
    axs[1].imshow(expected_img, cmap='viridis')
    axs[1].set_title("Expected Output")
    axs[2].imshow(prediction_img, cmap='viridis')
    axs[2].set_title("Prediction")
    axs[3].imshow(heatmap, cmap='gray')
    axs[3].set_title("Heatmap (Correct = 1)")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    filename = f"prediction_debug_{model_index}.png"
    plt.savefig(filename)
    plt.close()
    log(f"[INFO] Debug visual salvo: {filename}")