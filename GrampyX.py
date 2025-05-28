

# GrampyX_persistente.py
import tensorflow as tf
import numpy as np
import os
import pickle
import random
import json
from court_logic import arc_court_supreme
from confidence_system import ConfidenceManager
from metrics_utils import plot_prediction_test, gerar_video_time_lapse, embutir_trilha_sonora
from runtime_utils import save_debug_result
from metrics_utils import log
from models_loader import load_model
from data_pipeline import load_data_batches
from train_all import training_process
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from GrampyX_internal_models import GrumpyInternalModels

PERSIST_DIR = "grampy_data"
DETECTOR_WEIGHTS = os.path.join(PERSIST_DIR, "detector.h5")
HISTORY_PATH = os.path.join(PERSIST_DIR, "history.pkl")

class GrampyX:
    def __init__(self, num_modelos=7):
        os.makedirs(PERSIST_DIR, exist_ok=True)

        self.num_modelos = num_modelos
        self.models = [load_model(i, 0.0005) for i in range(num_modelos)]
        self.manager = ConfidenceManager(self.models)
        self.submission_dict = []
        self.history_X = []
        self.history_y = []
        self.internal_models = GrumpyInternalModels()

        self.detector = Sequential([
            Input(shape=(30 * 30,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.detector.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        if os.path.exists(DETECTOR_WEIGHTS):
            self.detector.load_weights(DETECTOR_WEIGHTS)
            log("[GrampyX] Pesos do detector carregados")

        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "rb") as f:
                self.history_X, self.history_y = pickle.load(f)
            log("[GrampyX] Histórico carregado")

    def salvar_estado(self):
        self.detector.save_weights(DETECTOR_WEIGHTS)
        with open(HISTORY_PATH, "wb") as f:
            pickle.dump((self.history_X, self.history_y), f)
        log("[GrampyX] Estado salvo")

    def preparar_inputs(self, x):
        if x.shape[-1] != 40:
            log(f"[WARN] Input com shape errado: {x.shape}, ajustando para (1, 30, 30, 10, 40)")
            # Preenche o restante com zeros
            pad = tf.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], 40 - x.shape[-1]), dtype=x.dtype)
            x = tf.concat([x, pad], axis=-1)
        return x, x


    def julgar(self, x_train, y_train, y_val, x_input, raw_input, block_index, task_id, idx, iteracao, Y_val):
        try:
            x_input = tf.expand_dims(tf.convert_to_tensor(x_input, dtype=tf.float32), axis=0)
            x_outros, _ = self.preparar_inputs(x_input)

            log(f"[DEBUG] Shape final input para modelos: {x_outros.shape}")
            log(f"[GrampyX] Julgando bloco {block_index} — Task {task_id}")

            resultados = arc_court_supreme(
                models=self.models,
                X_train=x_train,
                y_train=y_train,
                y_val=y_val,
                X_test=x_outros,
                task_id=task_id,
                block_idx=block_index,
                confidence_manager=self.manager,
                idx=idx,
                Y_val=None
            )

            consenso = resultados.get("consenso", 0.0)
            y_pred = resultados.get("class_logits") if isinstance(resultados, dict) else resultados

            if isinstance(y_pred, tf.Tensor):
                try:
                    flat = tf.argmax(y_pred, axis=-1).numpy().flatten()
                    if flat.shape[0] != 9000:
                        raise ValueError(f"[ERRO] Shape inesperado após argmax: {flat.shape}")
                    label = 1.0 if consenso >= 0.9 else 0.0
                    self.history_X.append(flat)
                    self.history_y.append(label)
                except Exception as e:
                    log(f"[GrampyX ERRO] Erro ao preparar histórico: {e}")
            else:
                log("[GrampyX WARN] y_pred não é tensor. Histórico não será atualizado.")

            self.internal_models.adicionar_voto(y_pred.numpy() if isinstance(y_pred, tf.Tensor) else y_pred, consenso)
            self.internal_models.treinar_todos()

            if len(self.history_X) >= 10:
                try:
                    self.detector.fit(
                        np.array(self.history_X),
                        np.array(self.history_y),
                        epochs=5,
                        verbose=0
                    )
                    log("[GrampyX] Detector interno treinado")
                except Exception as e:
                    log(f"[GrampyX ERRO] Falha ao treinar detector: {e}")

            self.salvar_estado()

            salvar_path = f"images/clippy/JULGAMENTO_{block_index}_{task_id}"
            plot_prediction_test(raw_input=raw_input, predicted_output=y_pred, save_path=salvar_path, pad_value=0)

            self.submission_dict.append({"task_id": task_id, "prediction": y_pred})
            save_debug_result(self.submission_dict, "submission.json")

            video_path = gerar_video_time_lapse("votos_visuais", block_index, f"{block_index}_{task_id}.avi")
            if video_path:
                embutir_trilha_sonora(video_path, block_index)

            return {"consenso": consenso}

        except Exception as e:
            log(f"[GrampyX ERRO] Bloco {block_index}: {str(e)}")
            return {"consenso": 0.0}



# Global cache para manter batches entre chamadas
todos_os_batches = {}
def extrair_classes_validas(y_real, pad_value=0):
    y_real = tf.convert_to_tensor(y_real)
    log(f"[DEBUG] extrair_classes_validas — y_real.shape={y_real.shape}")

    # Se a forma for (H, W, 1, 4) ou (30, 30, 1, 4), extrai canal de cor
    if y_real.shape.rank == 4 and y_real.shape[-1] == 4:
        y_real = y_real[..., 0]  # Pega apenas o canal da classe

    # Remover dimensão -1 apenas se ela for 1
    if y_real.shape.rank >= 4 and y_real.shape[-1] == 1:
        y_real = tf.squeeze(y_real, axis=-1)
    elif y_real.shape.rank >= 4 and y_real.shape[-1] != 1:
        log(f"[WARN] Tentativa de squeeze em shape incompatível: {y_real.shape}")

    valores = tf.unique(tf.reshape(y_real, [-1]))[0]
    valores = tf.cast(valores, tf.int32)
    valores_validos = tf.boolean_mask(valores, valores != pad_value)

    log(f"[DEBUG] Valores únicos: {valores.numpy().tolist()}")
    log(f"[DEBUG] Classes extraídas: {valores_validos.numpy().tolist()}")
    return valores_validos







def rodar_deliberacao_com_condicoes(parar_se_sucesso=True, max_iteracoes=100, consenso_minimo=0.9, idx=0, grampyx=None):
    grampy = grampyx
    with open("arc-agi_test_challenges.json") as f:
        test_challenges = json.load(f)
    task_ids = list(test_challenges.keys())
    # clippy.models = []
    # clippy.models = [load_model(i, 0.0005) for i in range(clippy.num_modelos)]
    if idx not in todos_os_batches:
        todos_os_batches[idx] = []

        for model_idx in range(grampy.num_modelos):
            batches = load_data_batches(
                challenges=test_challenges,
                num_models=grampy.num_modelos,
                task_ids=task_ids,
                model_idx=model_idx,
                block_idx=idx
            )
            training_process(
                models=grampy.models,
                batches=batches,
                n_model=model_idx,
                batch_index=idx,
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
            )
            todos_os_batches[idx].extend(batches)
    sucesso_global = False

    for (X_train, X_val, Y_train, Y_val, X_test, raw_input, block_idx, task_id) in todos_os_batches[idx]:
        iteracao = 0
        sucesso = False

        while not sucesso and iteracao < max_iteracoes:
            log(f"[GrampyX] Deliberação iter {iteracao} — Task {task_id} — Bloco {block_idx}")
            y_val_test = extrair_classes_validas(X_test, 0)
            log(f"[GRAMPYX] y_val_test: {y_val_test}")
            resultado = grampy.julgar(x_train=X_train, y_train=Y_train, y_val=Y_val,
                x_input=X_test, raw_input=raw_input, block_index=block_idx, task_id=task_id, idx=idx, iteracao=iteracao, Y_val=y_val_test )

            consenso = resultado.get("consenso", 0)
            if consenso >= consenso_minimo:
                log(f"[GrampyX] Consenso alcançado ({consenso:.2f}), encerrando iteração.")
                sucesso = True
                sucesso_global = True
                break
            else:
                log(f"[GrampyX] Consenso insuficiente ({consenso:.2f}), nova rodada.")
                iteracao += 1

        if not sucesso and parar_se_sucesso:
            log(f"[GrampyX] Máximo de iterações atingido para bloco {block_idx}. Partindo pro próximo.")

    log("[GrampyX] Deliberação encerrada.")
    return sucesso_global
