from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf

VOCAB_SIZE = 10

def compute_aggressive_class_weights(y_train, agressiveness=1.0):
    y_train_flat = y_train.numpy().flatten()
    classes = np.unique(y_train_flat)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_flat)

    # Penaliza menos agressivamente a classe 0
    weights[0] *= agressiveness # em vez de obliterar

    class_weight_array = np.ones(VOCAB_SIZE)
    for cls, weight in zip(classes, weights):
        class_weight_array[cls] = weight

    return class_weight_array



def spatial_augmentations(image_tensor, label_tensor):
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)

    # Garante que o rótulo tenha shape (H, W, 1) para rotação
    label_tensor = tf.expand_dims(label_tensor, axis=-1)
    image_tensor = tf.image.rot90(image_tensor, k)
    label_tensor = tf.image.rot90(label_tensor, k)

    # Remove canal extra para voltar ao shape (H, W)
    label_tensor = tf.squeeze(label_tensor, axis=-1)

    return image_tensor, label_tensor


