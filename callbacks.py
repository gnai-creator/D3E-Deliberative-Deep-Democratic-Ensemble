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
        pred_logits = self.model.predict(self.x_val, verbose=0)["main_output"]
        metrics = compute_metrics(self.y_val, pred_logits, self.num_classes)
        
        log("\n[Custom Metrics]")
        log(f"  Mean Accuracy: {metrics['mean_accuracy']:.3f}")
        log(f"  Mean IoU: {metrics['mean_iou']:.3f}")
        log(f"  Entropy: {metrics['entropy']:.3f}")
        for cls, acc in metrics["per_class_accuracy"].items():
            log(f"  Acc {cls}: {acc:.3f}")
        for cls, iou in metrics["per_class_iou"].items():
            log(f"  IoU {cls}: {iou:.3f}")
