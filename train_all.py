
import os
import json
import time
import random
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from metrics_utils import plot_prediction_debug
from runtime_utils import log, to_numpy_safe
from data_loader import load_data
from models_loader import load_model



def corte_esta_completa(models):
    return isinstance(models, list) and len(models) == 5 and all(m is not None for m in models)


def training_process(
    batches=[],
    models=[],
    n_model=0,
    batch_index=0,
    max_blocks=1,
    block_size=1,
    max_training_time=14400,
    cycles=150,
    epochs=60,
    batch_size=8,
    patience=20,
    rl_lr=2e-3,
    factor=0.65,
    len_trainig=1,
    pad_value=0,
):
    start_time = time.time()
    blacklist_blocks = [73, 124]

    if batch_index in blacklist_blocks:
        log(f"[SKIP] Bloco {batch_index} na blacklist.")
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

    _ = model(X_train, training=False)

   
    for cycle in range(cycles):
        log(f"Cycle {cycle} — Modelo {n_model}")
        model.fit(
            x=X_train,
            y=Y_train,
            validation_data=(X_val, Y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                # EarlyStopping(monitor="val_shape_acc", patience=patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=factor, patience=2, min_lr=rl_lr),
            ],
            verbose=0,
        )

        try:
            preds = model.predict(X_val)


            log(f"[DEBUG] preds shape: {preds.shape}")
            log(f"[DEBUG] y_val shape: {Y_val.shape}")
            # Se for logits de 10 classes, aplica argmax
            if preds.shape[-1] == 10:
                preds = np.argmax(preds, axis=-1)  # shape → (1, 30, 30, 1)
            # Remove batch e dimensão do canal se necessário

            log(f"[DEBUG] preds shape: {preds.shape}")
            log(f"[DEBUG] y_val shape: {Y_val.shape}")

            pixel_color_perfect, pixel_shape_perfect = plot_prediction_debug(
                raw_input=X_train,
                expected_output=Y_val,
                predicted_output=preds,
                model_index=f"block_{batch_index}_task_{task_id}_model_{n_model}",
                pad_value=pad_value,
                index=batch_index,
                task_id=task_id,
            )

            log(
                f"🎯 Pixel Color Perfect: {pixel_color_perfect:.5f} | Shape Perfect: {pixel_shape_perfect:.5f}"
            )

            if pixel_color_perfect >= 0.999 and pixel_shape_perfect >= 0.999:
                log(f"[✓] Modelo {n_model} treinado com sucesso no exemplo {batch_index}.")
                return

        except Exception as e:
            log(f"[ERROR] Erro ao avaliar bloco {batch_index}: {e}")
            traceback.print_exc()
