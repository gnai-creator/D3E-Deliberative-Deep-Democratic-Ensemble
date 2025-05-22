from metrics import compute_metrics
import tensorflow as tf
from runtime_utils import log

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, num_classes):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, logs=None):
        # 🔒 Garante que batch sizes sejam compatíveis
        x_val = self.x_val
        y_val = self.y_val
        if tf.shape(x_val)[0] != tf.shape(y_val)[0]:
            x_val = x_val[:1]
            y_val = y_val[:1]

        pred_logits = self.model.predict(x_val, verbose=0)["main_output"]
        metrics = compute_metrics(y_val, pred_logits, self.num_classes)

        log("\n[Custom Metrics]")
        log(f"  Mean Accuracy: {metrics['mean_accuracy']:.3f}")
        log(f"  Mean IoU: {metrics['mean_iou']:.3f}")
        log(f"  Entropy: {metrics['entropy']:.3f}")
        for cls, acc in metrics["per_class_accuracy"].items():
            log(f"  Acc {cls}: {acc:.3f}")
        for cls, iou in metrics["per_class_iou"].items():
            log(f"  IoU {cls}: {iou:.3f}")

# Regularização manual da matriz após cada batch — Callback style
class PermutationRegularizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight

    def on_train_batch_end(self, batch, logs=None):
        color_layer = getattr(self.model, "learned_color_permutation", None)

        if color_layer and hasattr(color_layer, "permutation_weights"):
            matrix = color_layer.permutation_weights
            identity = tf.eye(tf.shape(matrix)[0])
            reg_loss = tf.reduce_sum(tf.square(matrix - identity))
            self.model.add_loss(lambda: self.weight * reg_loss)


class EnableRefinerCallback(tf.keras.callbacks.Callback):
    def __init__(self, enable_epoch=5):
        super().__init__()
        self.enable_epoch = enable_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, 'use_refiner'):
            if epoch >= self.enable_epoch:
                if not self.model.use_refiner:
                    tf.print(f"[Callback] Refiner ativado na época {epoch}")
                    self.model.use_refiner.assign(True)
            else:
                if self.model.use_refiner:
                    tf.print(f"[Callback] Refiner desativado na época {epoch}")
                    self.model.use_refiner.assign(False)

