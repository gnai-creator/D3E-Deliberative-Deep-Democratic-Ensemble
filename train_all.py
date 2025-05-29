import os
import json
import time
import random
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from metrics_utils import plot_prediction_debug
from runtime_utils import log, to_numpy_safe
from data_loader import load_data
from models_loader import load_model

def corte_esta_completa(models):
    return isinstance(models, list) and len(models) == 5 and all(m is not None for m in models)


def prepare_validation_data(X_val, Y_val, pad_value=-1):
    # Garante tipos
    X_val = tf.cast(X_val, tf.float32)
    Y_val = tf.cast(Y_val, tf.int32)

    # Remove eixos desnecessários
    if X_val.shape.rank == 5 and X_val.shape[0] == 1:
        X_val = tf.squeeze(X_val, axis=0)  # (30, 30, 1, 1)
    if Y_val.shape.rank == 4 and Y_val.shape[0] == 1:
        Y_val = tf.squeeze(Y_val, axis=0)  # (30, 30, 1)

    if X_val.shape.rank == 4 and X_val.shape[-1] == 1 and X_val.shape[-2] == 1:
        X_val = tf.squeeze(X_val, axis=-1)  # (30, 30, 1)
        X_val = tf.squeeze(X_val, axis=-1)  # (30, 30)

    if Y_val.shape.rank == 3 and Y_val.shape[-1] == 1:
        Y_val = tf.squeeze(Y_val, axis=-1)  # (30, 30)

    # Máscara válida
    mask = tf.not_equal(Y_val, pad_value)  # (30, 30)

    # Flatten
    X_val_flat = tf.reshape(X_val, [-1, 1, 1, 1, 1])  # (900, 1, 1, 1, 1)
    Y_val_flat = tf.reshape(Y_val, [-1])              # (900,)
    mask_flat = tf.reshape(mask, [-1])                # (900,)

    # Aplica a máscara
    X_val_masked = tf.boolean_mask(X_val_flat, mask_flat)
    Y_val_masked = tf.boolean_mask(Y_val_flat, mask_flat)

    return X_val_masked, Y_val_masked



def training_process(
    batches=[],
    models=[],
    n_model=0,
    batch_index=0,
    max_blocks=1,
    block_size=1,
    max_training_time=14400,
    cycles=150,
    epochs=100,
    batch_size=4,
    patience=20,
    rl_lr=2e-3,
    factor=0.65,
    len_trainig=1,
    pad_value=-1,
):
    start_time = time.time()
    blacklist_blocks = [73, 124]

    if batch_index in blacklist_blocks:
        log(f"[SKIP] Bloco {batch_index} na blacklist.")
        return

    if not batches or batch_index >= len(batches):
        log(f"[ERRO] Nenhum batch carregado no índice {batch_index}.")
        return

    (
        X_train,
        X_val,
        Y_train,
        Y_val,
        X_test,
        raw_input,
        block_idx,
        task_id,
    ) = batches[batch_index]

    if n_model >= 5:
        log(f"[SKIP] Índice {n_model} fora da lista de modelos ({len(models)})")
        return

    model = models[n_model]
    if model is None:
        raise ValueError(f"[FATAL] Modelo {n_model} não foi carregado corretamente.")

    log(f"[INFO] Treinando modelo {n_model} no exemplo {batch_index}")
    log(f"[INFO] SHAPE X TRAIN : {X_train.shape}")
    log(f"[INFO] SHAPE Y TRAIN : {Y_train.shape}")

    # X_train = tf.cast(X_train, tf.float32)
    # X_train = tf.where(tf.less(X_train, 0), 0.0, X_train)  # substitui -1 por 0
    Y_train = tf.cast(Y_train, tf.int32)  # Sem tf.where

    X_val_masked, Y_val_masked = prepare_validation_data(X_val, Y_val)

    visual_grid = tf.squeeze(Y_train)  # reduz todas as dimensões de tamanho 1

    # Se ainda tiver mais de 2 dimensões, selecione o canal 0
    if visual_grid.shape.rank > 2:
        visual_grid = visual_grid[..., 0]  # garante shape (30, 30)

    plt.imshow(visual_grid.numpy(), cmap="viridis")
    plt.title("Input Visual")
    plt.colorbar()
    plt.savefig("a.png")
    _ = model(X_train, training=False)

    for cycle in range(cycles):
        log(f"Cycle {cycle} — Modelo {n_model}")
        try:

            checkpoint_callback = ModelCheckpoint(
                filepath=f"checkpoint_model_{n_model}_block_{batch_index}.h5",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
            
            model.fit(
                x=X_train,
                y=Y_train,
                validation_data=(X_val_masked, Y_val_masked),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[
                    ReduceLROnPlateau(monitor="val_loss", factor=factor, patience=2, min_lr=rl_lr),
                    EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
                    checkpoint_callback,
                ],
                verbose=1,
            )
        except Exception as e:
            log(f"[ERROR DETECTADO] {e}")
            traceback.print_exc()

        try:
            preds = model.predict(X_val)

            log(f"[DEBUG] preds shape: {preds.shape}")
            log(f"[DEBUG] y_val shape: {Y_val.shape}")

            if preds.shape[-1] == 10:
                preds = np.argmax(preds, axis=-1)

            # Arredonda para baixo a grade 30x30x1 e limita entre 0 e 9
            float_block = preds[0, :, :, :, 0]                     # (30, 30, 1)
            rounded = tf.floor(float_block + 0.5)                  # Arredonda corretamente
            clipped = tf.clip_by_value(rounded, 0, 9)              # Limita entre 0 e 9
            preds[0, :, :, :, 0] = tf.cast(clipped, tf.int32)


            pixel_color_perfect, pixel_shape_perfect = plot_prediction_debug(
                raw_input=X_train,
                expected_output=Y_val,
                predicted_output=preds,
                model_index=f"block_{batch_index}_task_{task_id}_model_{n_model}",
                pad_value=pad_value,
                index=batch_index,
                task_id=task_id,
            )

            log(f"🎯 Pixel Color Perfect: {pixel_color_perfect:.5f} | Shape Perfect: {pixel_shape_perfect:.5f}")

            if pixel_color_perfect >= 0.999 and pixel_shape_perfect >= 0.999:
                log(f"[✓] Modelo {n_model} treinado com sucesso no exemplo {batch_index}.")
                return

        except Exception as e:
            log(f"[ERROR] Erro ao avaliar bloco {batch_index}: {e}")
            traceback.print_exc()
