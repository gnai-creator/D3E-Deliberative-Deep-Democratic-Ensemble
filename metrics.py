import tensorflow as tf

def per_class_accuracy(y_true, y_pred, num_classes):
    """
    Calcula a acurácia individual para cada classe.
    Args:
        y_true: tensor [B, H, W] ou [N]
        y_pred: tensor [B, H, W] ou [N]
    Returns:
        List of tuples (class_index, accuracy)
    """
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    accs = []
    for c in range(num_classes):
        mask = tf.equal(y_true, c)
        correct = tf.reduce_sum(tf.cast(tf.logical_and(mask, tf.equal(y_pred, c)), tf.float32))
        total = tf.reduce_sum(tf.cast(mask, tf.float32))
        acc = tf.math.divide_no_nan(correct, total)
        accs.append((c, acc))
    return accs


def iou_score(y_true, y_pred, num_classes):
    """
    Calcula Intersection over Union por classe.
    """
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    ious = []
    for c in range(num_classes):
        y_true_c = tf.cast(tf.equal(y_true, c), tf.float32)
        y_pred_c = tf.cast(tf.equal(y_pred, c), tf.float32)
        intersection = tf.reduce_sum(y_true_c * y_pred_c)
        union = tf.reduce_sum(tf.clip_by_value(y_true_c + y_pred_c, 0, 1))
        iou = tf.math.divide_no_nan(intersection, union)
        ious.append((c, iou))
    return ious


def prediction_entropy(logits):
    """
    Calcula a entropia média dos logits (softmax).
    Args:
        logits: [B, H, W, C]
    Returns:
        Entropia média (float)
    """
    probs = tf.nn.softmax(logits, axis=-1)
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-9), axis=-1)
    return tf.reduce_mean(entropy)


def compute_metrics(y_true, y_pred_logits, num_classes, pad_value=-1):
    """
    Wrapper geral para computar métricas com máscara de padding e ignora classe 0 nas métricas de cor.
    
    Args:
        y_true: [B, H, W] int
        y_pred_logits: [B, H, W, C] logits (não softmax)
        num_classes: número total de classes
        pad_value: valor usado como padding no y_true (ex: -1)

    Returns:
        dict com métricas
    """
    # Converte logits para predições discretas
    y_pred = tf.argmax(y_pred_logits, axis=-1, output_type=tf.int32)

    # Cria máscara para ignorar regiões com padding
    valid_mask = tf.not_equal(y_true, pad_value)

    # Ignora também a classe 0 (só importa forma, não cor)
    class_mask = tf.not_equal(y_true, 0)
    combined_mask = tf.logical_and(valid_mask, class_mask)

    y_true_masked = tf.boolean_mask(y_true, combined_mask)
    y_pred_masked = tf.boolean_mask(y_pred, combined_mask)

    # Calcula métricas usando apenas regiões válidas e não-classe-0
    accs = per_class_accuracy(y_true_masked, y_pred_masked, num_classes)
    ious = iou_score(y_true_masked, y_pred_masked, num_classes)
    entropy_val = prediction_entropy(y_pred_logits)

    # Filtra para excluir classe 0 nos dicionários
    accs = [a for a in accs if a[0] != 0]
    ious = [i for i in ious if i[0] != 0]

    # Monta dicionário de métricas
    metrics = {
        "mean_accuracy": tf.reduce_mean([a[1] for a in accs]),
        "mean_iou": tf.reduce_mean([i[1] for i in ious]),
        "entropy": entropy_val,
        "per_class_accuracy": {f"class_{a[0]}": a[1].numpy() for a in accs},
        "per_class_iou": {f"class_{i[0]}": i[1].numpy() for i in ious},
    }

    return metrics
