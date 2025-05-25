import os
import json
import time
import math
import random
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from metrics_utils import plot_prediction_debug, plot_prediction_test, gerar_video_time_lapse, embutir_trilha_sonora
from runtime_utils import log, save_debug_result, transform_input, to_numpy_safe
from data_preparation import get_dataset
from model_loader import load_model
from court_logic import arc_court_supreme

def corte_esta_completa(models):
    return isinstance(models, list) and len(models) == 5 and all(m is not None for m in models)

def test_challenge(models, X_test, model_idx, raw_test_inputs, block_index, task_id, submission_dict):
    log(f"[TEST] Inicializando teste bloco {block_index} para task {task_id}")

    try:
        # Verifica entrada
        if X_test is None or tf.size(X_test) == 0 or raw_test_inputs is None or len(raw_test_inputs) == 0:
            raise ValueError("[TEST] Entrada de teste vazia ou malformada.")

        # Prepara tensor de entrada
        x_test_sample = tf.convert_to_tensor(X_test[0], dtype=tf.float32)
        x_test_sample = tf.expand_dims(x_test_sample, axis=0)  # shape (1, H, W, C)

        # Se tiver 40 canais, já é entrada do juiz
        if x_test_sample.shape[-1] == 40:
            x_input_juiz = x_test_sample
            x_input_outros = tf.concat(tf.split(x_test_sample, num_or_size_splits=10, axis=-1)[:1], axis=-1)
        else:
            x_input_outros = x_test_sample
            x_input_juiz = tf.zeros((1, 30, 30, 1, 40), dtype=tf.float32)

        # Prepara os modelos (certifica que são 6)
        if not models or len(models) < 5:
            raise ValueError("[TEST] Modelos insuficientes para formar corte inicial.")

        # Garante modelo da Suprema Juíza
        if len(models) < 6:
            suprema = load_model(5, 0.0005)
            models.append(suprema)
            log("[TEST] Modelo da Suprema Juíza adicionado à corte.")
        elif len(models) > 6:
            models = models[:6]
            log("[TEST] Lista de modelos truncada para 6 membros.")

        # Sanidade
        if len(models) != 6:
            raise ValueError(f"[TEST] Corte incorreta: esperados 6 modelos, recebidos {len(models)}")

        log(f"[TEST] Preview raw input: {np.unique(raw_test_inputs[0])}")

        # Julgamento
        preds = arc_court_supreme(models, x_input_outros, task_id=task_id, block_idx=block_index)

        # Gera vídeo
        video_path = gerar_video_time_lapse(pasta="votos_visuais", block_idx=block_index, saida=f"{block_index}_{task_id}.avi")
        if video_path:
            embutir_trilha_sonora(video_path=video_path, block_idx=block_index)

        # Pós-processamento
        y_test_logits = preds["class_logits"] if isinstance(preds, dict) else preds
        pred_np = y_test_logits.numpy() if isinstance(y_test_logits, tf.Tensor) else y_test_logits

        log(f"[DEBUG] predicted_output shape before plot: {pred_np.shape}")

        plot_prediction_test(
            raw_input=raw_test_inputs[0],
            predicted_output=pred_np,
            save_path=f"images/test/TEST_block_{block_index}_task_{task_id}_model_{model_idx}",
            pad_value=PAD_VALUE
        )

        plot_prediction_test(
            raw_input=raw_test_inputs[0],
            predicted_output=pred_np,
            save_path=f"images/test/PREDICT TEST JUDGE RESULTS AAASSSSXXX",
            pad_value=PAD_VALUE
        )

        submission_dict.append({
            "task_id": task_id,
            "prediction": pred_np
        })
        save_debug_result(submission_dict, "submission.json")

        log(f"[TEST] Teste concluído para task {task_id}, bloco {block_index}")

    except Exception as e:
        log(f"[ERRO] Exceção durante test_challenge: {str(e)}")
        raise

        log(f"[ERROR] Erro ao gerar predicoes: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Minimiza logs (1: INFO, 2: WARNING, 3: ERROR)

    PAD_VALUE = 0
    VOCAB_SIZE = 10
    LEARNING_RATE = 0.0005
    PATIENCE = 20
    RL_LEARNING_RATE = 2e-3
    FACTOR = 0.65
    BATCH_SIZE = 8
    EPOCHS = 60
    RESULTS_DIR = "results"
    BLOCK_SIZE = 1
    LEN_TRAINING = 1
    CYCLES = 150
    MAX_BLOCKS = 400
    MAX_TRAINING_TIME = 4 * 60 * 60
    N_MODELS = 5

    with open("arc-agi_test_challenges.json") as f:
        test_challenges = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    task_ids = list(test_challenges.keys())
    start_time = time.time()
    submission_dict = []
    blacklist_blocks = [73, 124]

    for _ in range(LEN_TRAINING):
        block_index = 0

        while block_index < MAX_BLOCKS and time.time() - start_time < MAX_TRAINING_TIME:
            if block_index in blacklist_blocks:
                block_index += 1
                continue

            model_idx = 0
            models = []
            log(f"Treinando bloco {block_index:02d}")

            while model_idx < 5:
                X_train, X_val, Y_train, Y_val, _, _, X_test, info_train, info_val, task_id, raw_inputs, raw_test_inputs = get_dataset(
                    block_index=block_index, task_ids=task_ids, challenges=test_challenges,
                    block_size=BLOCK_SIZE, pad_value=PAD_VALUE, vocab_size=VOCAB_SIZE,
                    model_idx=model_idx
                )

                log(f"[INFO] SHAPE X TRAIN : {X_train.shape}")
                log(f"[INFO] SHAPE Y Val : {Y_val.shape}")
                log(f"[INFO] SHAPE Y TRAIN : {Y_train.shape}")

                model = load_model(model_idx, LEARNING_RATE)
                if model is None:
                    raise ValueError(f"[FATAL] Modelo {model_idx} não foi carregado corretamente.")

                if model_idx == 4:
                    Y_train = tf.squeeze(Y_train, axis=-1)
                    Y_val = tf.squeeze(Y_val, axis=-1)

                _ = model(X_train, training=False)

                for cycle in range(CYCLES):
                    log(f"Cycle {cycle} MODEL : {model_idx}")
                    # log(f"X_train.shape: {X_train.shape}")
                    # log(f"Y_train.shape: {Y_train.shape}")

                    model.fit(
                        x=X_train,
                        y=Y_train,
                        validation_data=(X_val, Y_val),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[
                            EarlyStopping(monitor="val_shape_acc", patience=PATIENCE, restore_best_weights=True),
                            ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=2, min_lr=RL_LEARNING_RATE)
                        ],
                        verbose=0
                    )

                    try:
                        x_val_sample = transform_input(X_val[:1])
                        preds = model.predict(x_val_sample)
                        y_val_logits = preds["class_logits"] if isinstance(preds, dict) else preds
                        y_val_pred = tf.argmax(y_val_logits, axis=-1).numpy()[0]
                        y_val_expected = to_numpy_safe(Y_val[:1][0])

                        valid_mask = (y_val_expected != PAD_VALUE)

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
                            raw_input=raw_inputs[0],
                            expected_output=y_val_expected,
                            predicted_output=y_val_pred,
                            model_index=f"block_{block_index}_task_{task_id}_model_{model_idx}",
                            pad_value=PAD_VALUE,
                            index=block_index,
                            task_id=task_id
                        )
                        plot_prediction_debug(
                            raw_input=raw_inputs[0],
                            expected_output=y_val_expected,
                            predicted_output=y_val_pred,
                            model_index=f"AAASSSSXXX",
                            pad_value=PAD_VALUE,
                            index=block_index,
                            task_id=task_id
                        )

                        log(f"🎯 Pixel Color Perfect: {pixel_color_perfect:.5f} - Pixel Shape Perfect: {pixel_shape_perfect:.5f}")

                        if pixel_color_perfect >= 0.999 and pixel_shape_perfect >= 0.999:
                            if model not in models:
                                models.append(model)

                            if corte_esta_completa(models):
                                test_challenge(models, X_test, model_idx, raw_test_inputs, block_index, task_id, submission_dict)
                                block_index += 1
                                model_idx += 1
                                break

                            model_idx += 1
                            break

                    except Exception as e:
                        log(f"[ERROR] Erro ao avaliar bloco {block_index}: {e}")
                        traceback.print_exc()
