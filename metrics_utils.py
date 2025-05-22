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


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from runtime_utils import log


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from runtime_utils import log


def plot_prediction_debug(input_tensor, expected_output, predicted_output, model_index, index):
    def to_numpy_safe(tensor):
        """Converte tensor para numpy e valida tipo."""
        if isinstance(tensor, tf.Tensor):
            tensor = tensor.numpy()
        tensor = np.asarray(tensor)
        if tensor.dtype.kind in {'U', 'S', 'O'}:
            raise TypeError(f"[ERROR] Tensor com tipo inválido: {tensor.dtype}")
        return tensor

    def argmax_if_logits(tensor):
        """Aplica argmax se tensor tiver canal de classe."""
        return np.argmax(tensor, axis=-1) if tensor.ndim >= 3 and tensor.shape[-1] > 1 else tensor

    try:
        # Conversões seguras
        input_tensor = to_numpy_safe(input_tensor)
        expected_output = to_numpy_safe(expected_output).astype(np.int32)
        predicted_output = to_numpy_safe(predicted_output)
        predicted_output = argmax_if_logits(predicted_output).astype(np.int32)

        # Processa input_img
        if input_tensor.ndim == 4:  # [H, W, T, C]
            input_img = argmax_if_logits(input_tensor[:, :, 0, :])
        elif input_tensor.ndim == 3:
            input_img = argmax_if_logits(input_tensor)
        elif input_tensor.ndim == 2:
            input_img = input_tensor
        else:
            raise ValueError(f"[ERROR] input_tensor shape inválido: {input_tensor.shape}")

        # Corrige shape da predição se necessário
        if predicted_output.ndim == 1 and expected_output.ndim == 2:
            if predicted_output.size == expected_output.shape[0]:
                predicted_output = np.tile(predicted_output[:, None], (1, expected_output.shape[1]))
            elif predicted_output.size == expected_output.shape[1]:
                predicted_output = np.tile(predicted_output[None, :], (expected_output.shape[0], 1))
            elif predicted_output.size == np.prod(expected_output.shape):
                predicted_output = predicted_output.reshape(expected_output.shape)
            else:
                raise ValueError("[ERROR] Shape de predição incompatível com o esperado.")

        # Validação final
        if not all(img.ndim == 2 for img in [input_img, expected_output, predicted_output]):
            raise ValueError("[ERROR] Todos os dados esperados no formato 2D (H, W).")

        # Gera heatmap
        heatmap = (predicted_output == expected_output).astype(np.int32)
        accuracy = np.mean(heatmap)

        # Plotagem
        fig, axs = plt.subplots(1, 4, figsize=(18, 4))
        titles = ["Input", "Expected Output", "Prediction", f"Accuracy ({accuracy:.1%})"]
        images = [input_img, expected_output, predicted_output, heatmap]
        cmaps = ["viridis", "viridis", "viridis", "gray"]

        for ax, img, title, cmap in zip(axs, images, titles, cmaps):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")

        plt.suptitle(f"Prediction Debug - Model {model_index}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"images/prediction_debug_{model_index}_index_{index}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        log(f"[INFO] Debug visual salvo: {filename}")

    except Exception as e:
        log(f"[ERROR] Falha ao gerar plot de debug: {e}")




def plot_raw_input_preview(tensor_raw, model_name):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np

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
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    import json

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


def ensure_numpy(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)







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