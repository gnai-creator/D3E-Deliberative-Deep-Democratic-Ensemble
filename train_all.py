
import os
import json
import time
import random
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from metrics_utils import plot_prediction_debug
from runtime_utils import log,  transform_input, to_numpy_safe
from data_loader import load_data
from models_loader import load_model



def corte_esta_completa(models):
    return isinstance(models, list) and len(models) == 5 and all(m is not None for m in models)


def training_process(
        batches=[],
        models=[],
        n_model=0,
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

    for _ in range(len_trainig):
        block_index = 0

        while block_index < max_blocks and time.time() - start_time < max_training_time:
            if block_index in blacklist_blocks:
                block_index += 1
                continue
            if block_index >= len(batches):
                 break


            model_idx = 0
            log(f"Treinando bloco {block_index:02d}")

            while model_idx < 5:
                X_train, X_val, Y_train, Y_val, X_test, raw_input, block_idx, task_id = batches[n_model]


                model = models[model_idx]
                if model is None:
                    raise ValueError(f"[FATAL] Modelo {model_idx} não foi carregado corretamente.")

                if model_idx == 4:
                    Y_train = tf.squeeze(Y_train, axis=-1)
                    Y_val = tf.squeeze(Y_val, axis=-1)
                
                if len(X_train.shape) == 4:
                    X_train = tf.expand_dims(X_train, axis=0)
                
                log(f"[INFO] SHAPE X TRAIN : {X_train.shape}")
                log(f"[INFO] SHAPE X Val : {X_val.shape}")
                log(f"[INFO] SHAPE Y TRAIN : {Y_train.shape}")
                log(f"[INFO] SHAPE Y Val : {Y_val.shape}")

                _ = model(X_train, training=False)

                for cycle in range(cycles):
                    log(f"Cycle {cycle} MODEL : {model_idx}")
                    

                    model.fit(
                        x=X_train,
                        y=Y_train,
                        validation_data=(X_val, Y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[
                            EarlyStopping(monitor="val_shape_acc", patience=patience, restore_best_weights=True),
                            ReduceLROnPlateau(monitor="val_loss", factor=factor, patience=2, min_lr=rl_lr)
                        ],
                        verbose=0
                    )

                    try:
                        x_val_sample = transform_input(X_val[:1])
                        preds = model.predict(x_val_sample)
                        y_val_logits = preds["class_logits"] if isinstance(preds, dict) else preds
                        y_val_pred = tf.argmax(y_val_logits, axis=-1).numpy()[0]
                        y_val_expected = to_numpy_safe(Y_val[:1][0])

                        valid_mask = (y_val_expected != pad_value)

                        if valid_mask.shape != y_val_pred.shape:
                            try:
                                valid_mask = np.squeeze(valid_mask)
                                if valid_mask.shape != y_val_pred.shape:
                                    raise ValueError("Shape incompatível após squeeze.")
                            except Exception:
                                log(f"[ERROR] Falha ao ajustar máscara: mask shape {valid_mask.shape}, pred shape {y_val_pred.shape}")
                                continue

                        if np.sum(valid_mask) == 0:
                            log(f"[WARN] Nenhum pixel válido para comparação. Task: {task_id}")
                            continue

                        pixel_color_perfect, pixel_shape_perfect = plot_prediction_debug(
                            raw_input=raw_input[0],
                            expected_output=y_val_expected,
                            predicted_output=y_val_pred,
                            model_index=f"block_{block_index}_task_{task_id}_model_{model_idx}",
                            pad_value=pad_value,
                            index=block_index,
                            task_id=task_id
                        )
                        plot_prediction_debug(
                            raw_input=raw_input[0],
                            expected_output=y_val_expected,
                            predicted_output=y_val_pred,
                            model_index=f"AAASSSSXXX",
                            pad_value=pad_value,
                            index=block_index,
                            task_id=task_id
                        )

                        log(f"🎯 Pixel Color Perfect: {pixel_color_perfect:.5f} - Pixel Shape Perfect: {pixel_shape_perfect:.5f}")

                        if pixel_color_perfect >= 0.999 and pixel_shape_perfect >= 0.999:
                            if corte_esta_completa(models):
                                return
                            block_idx += 1
                            model_idx += 1
                            break

                    except Exception as e:
                        log(f"[ERROR] Erro ao avaliar bloco {block_index}: {e}")
                        traceback.print_exc()
