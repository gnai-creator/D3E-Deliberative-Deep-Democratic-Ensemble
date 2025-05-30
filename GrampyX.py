

# GrampyX_persistente.py
import tensorflow as tf
import numpy as np
import os
import pickle
import random
import json
import traceback
from court_logic import arc_court_supreme
from court_utils import extrair_classes_validas, extrair_todas_classes_validas
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

def to_serializable(val):
    if isinstance(val, tf.Tensor):
        return val.numpy().tolist()
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, (np.int64, np.int32, np.float32, np.float64)):
        return val.item()
    return str(val)

def contar_blocos(challenges_path):
    import json
    with open(challenges_path) as f:
        challenges = json.load(f)
    return len(challenges)


class GrampyX:
    def __init__(self, num_modelos=7, challenges_path="arc-agi_test_challenges.json"):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        self.num_blocos = contar_blocos(challenges_path)  # nova função
        self.num_modelos = num_modelos
        self.models = []
        for model_index in range(num_modelos):
            tf.keras.utils.set_random_seed(42 + model_index)
            model = load_model(model_index, 0.0015)
            self.models.append(model)

     
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
        if x.shape.rank == 4:
            x = tf.expand_dims(x, axis=0)  # Garante (1, H, W, C, Conf)
        if x.shape[-1] != 1:
            pad = tf.zeros_like(x[..., :1])
            x = tf.concat([x, pad], axis=-1)
        return x, x

def julgar(self, x_train, y_train, y_val, x_input, raw_input, block_index, task_id, idx, iteracao, Y_val):
    try:
        log(f"[GrampyX] Julgando bloco {block_index} — Task {task_id}")
        log(f"[GrampyX] X_TRAIN SHAPE FINAL : {x_train.shape}")
        log(f"[GrampyX] Y_TRAIN SHAPE FINAL : {y_train.shape}")
        log(f"[GrampyX] Y_VAL SHAPE FINAL : {y_val.shape}")
        log(f"[GrampyX] X_TESTE SHAPE FINAL : {x_input.shape}")

        if x_input.shape.rank != 5 or x_input.shape != (1, 30, 30, 1, 1):
            x_input, _ = self.preparar_inputs(x_input)

        resultados = arc_court_supreme(
            models=self.models,
            X_train=x_train,
            y_train=y_train,
            y_val=y_val,
            X_test=x_input,
            task_id=task_id,
            block_idx=block_index,
            confidence_manager=self.manager,
            idx=idx,
            Y_val=Y_val
        )

        log(f"[DEBUG] Salvando arquivo com idx={idx}, iteracao={iteracao}, block_index={block_index}")

        consenso = float(resultados.get("consenso", 0.0))
        y_pred = resultados.get("class_logits")
        y_pred_simbolico = resultados.get("y_pred_simbolico")

        if isinstance(y_pred, tf.Tensor):
            try:
                flat = tf.argmax(y_pred, axis=-1).numpy().flatten()
                if flat.shape[0] != 900:
                    log(f"[GrampyX ERRO] Previsão descartada: shape inesperado após argmax: {flat.shape}")
                else:
                    label = 1.0 if consenso >= 0.9 else 0.0
                    self.history_X.append(flat)
                    self.history_y.append(label)
            except Exception as e:
                log(f"[GrampyX ERRO] Erro ao preparar histórico: {e}")
        else:
            log("[GrampyX WARN] y_pred não é tensor. Histórico não será atualizado.")

        # Valida classes previstas
        if y_pred_simbolico is not None:
            try:
                y_pred_tensor = tf.convert_to_tensor(y_pred_simbolico)
                classes_validas = extrair_todas_classes_validas(x_input, x_train, pad_value=-1)

                valores_preditos = tf.unique(tf.reshape(y_pred_tensor, [-1])).y
                set_preditos = set(valores_preditos.numpy().tolist())
                set_validas = set([int(c) for c in classes_validas])

                if set_preditos != set_validas:
                    extras = set_preditos - set_validas
                    faltando = set_validas - set_preditos
                    log(f"[AVALIAÇÃO] Classes previstas não batem.\n  → Extras: {extras}\n  → Faltando: {faltando}")
                    consenso = 0.0
                else:
                    log("[AVALIAÇÃO] Classes previstas batem exatamente com as válidas.")

            except Exception as e:
                log(f"[AVALIAÇÃO] Erro ao validar classes previstas: {e}")

        self.internal_models.adicionar_voto(y_pred.numpy() if isinstance(y_pred, tf.Tensor) else y_pred, consenso)
        self.internal_models.treinar_todos()

        if len(self.history_X) >= 10:
            try:
                tamanhos = set(map(len, self.history_X))
                if len(tamanhos) == 1:
                    self.detector.fit(
                        np.array(self.history_X),
                        np.array(self.history_y),
                        epochs=5,
                        verbose=0
                    )
                    log("[GrampyX] Detector interno treinado")
                else:
                    log("[GrampyX ERRO] history_X contém vetores com tamanhos diferentes: " + str(tamanhos))
            except Exception as e:
                log(f"[GrampyX ERRO] Falha ao treinar detector: {e}")

        self.salvar_estado()

        # Convertendo y_pred para valores inteiros entre 0 e 9 antes do submission
        if isinstance(y_pred, tf.Tensor):
            y_pred_np = y_pred.numpy()
            y_pred_np = y_pred_np.astype(np.float32)
            y_pred_np = np.floor(y_pred_np + 0.5)
            y_pred_np = np.clip(y_pred_np, 0, 9).astype(np.int32)
        else:
            y_pred_np = y_pred

        self.submission_dict.append({"task_id": task_id, "prediction": to_serializable(y_pred_np)})
        save_debug_result(self.submission_dict, "submission.json")

        return {"consenso": consenso, "y_pred_simbolico": y_pred_simbolico}

    except Exception as e:
        log(f"[GrampyX ERRO] Bloco {block_index}: {str(e)}")
        traceback.print_exc()
        return {"consenso": 0.0, "y_pred_simbolico": None}





# Global cache para manter batches entre chamadas
todos_os_batches = {}


def rodar_deliberacao_com_condicoes(parar_se_sucesso=True, max_iteracoes=100, consenso_minimo=9.5, idx=0, grampyx=None):
    grampy = grampyx
    BATCH_SIZE = 1
    block_idx = idx % grampy.num_blocos

    import os

    for model_idx in range(grampy.num_modelos):
        weights_path = f"weights_model_{model_idx}_block_{idx}.h5"
        if os.path.exists(weights_path):
            grampy.models[model_idx].load_weights(weights_path)
            log(f"[INFO] Pesos carregados: {weights_path}")
        else:
            log(f"[AVISO] Pesos não encontrados: {weights_path} — pulando carregamento.")

    with open("arc-agi_test_challenges.json") as f:
        test_challenges = json.load(f)

    task_ids = list(test_challenges.keys())

    if idx >= len(task_ids):
        log(f"[ERRO] Índice idx={idx} excede número de tarefas disponíveis ({len(task_ids)}). Abortando.")
        return False

    task_batch = [task_ids[idx]]

    # if idx not in todos_os_batches:
    todos_os_batches[idx] = []
    batches = []
    for model_idx in range(grampy.num_modelos):
        batches = load_data_batches(
            challenges=test_challenges,
            num_models=grampy.num_modelos,
            task_ids=task_batch,
            block_idx=0 if len(task_batch) == 1 else (idx % grampy.num_blocos)
        )
        if not batches:
            log(f"[ERRO] Nenhum batch retornado para task={task_batch}")
            return False

        training_process(
            models=grampy.models,
            batches=batches,
            n_model=model_idx,
            batch_index=0,
            max_blocks=1,
            block_size=1,
            max_training_time=14400,
            batch_size=8,          # ou até 4 se estiver lento demais
            epochs=40,             # mantém
            patience=10,            # para early stopping
            rl_lr=1.5e-3,            # ok
            factor=0.65,           # ok
            len_trainig=1,
            pad_value=-1,
        )
        todos_os_batches[idx].extend(batches)

    sucesso_global = False

    X_train, X_val, Y_train, Y_val, X_test, raw_input, block_idx, task_id = batches[0]
    iteracao = 0
    sucesso = False
    
    for model_idx in range(grampy.num_modelos):
        weights_path = f"weights_model_{model_idx}_block_{idx}.h5"
        if os.path.exists(weights_path):
            grampy.models[model_idx].load_weights(weights_path)
            log(f"[INFO] Pesos carregados: {weights_path}")
        else:
            log(f"[AVISO] Pesos não encontrados: {weights_path} — pulando carregamento.")

    while not sucesso and iteracao < max_iteracoes:
        log(f"[GrampyX] Deliberação iter {iteracao} — Task {task_id} — Bloco {block_idx}")
        y_val_test = extrair_classes_validas(X_test, 0)
        log(f"[GRAMPYX] y_val_test: {y_val_test}")
        print(f"[DEBUG] iteracao={iteracao}, block_idx={block_idx}, idx={idx}, task_id={task_id}")

        resultado = grampy.julgar(
            x_train=X_train,
            y_train=Y_train,
            y_val=Y_val,
            x_input=X_test,
            raw_input=raw_input,
            block_index=block_idx,
            task_id=task_id,
            idx=iteracao,
            iteracao=iteracao,
            Y_val=y_val_test
        )

        consenso = float(resultado.get("consenso", 0))
        
        y_pred = resultado.get("y_pred_simbolico")
        if y_pred is not None:
            try:
                y_pred_tensor = tf.convert_to_tensor(y_pred)
                classes_validas = extrair_todas_classes_validas(X_test, X_train, pad_value=-1)

                # Extrai valores únicos previstos
                valores_preditos = tf.unique(tf.reshape(y_pred_tensor, [-1])).y

                # Corrige aqui: converte para lista antes de transformar em set
                set_preditos = set(valores_preditos.numpy().tolist())
                set_validas = set(classes_validas)

                if set_preditos != set_validas:
                    extras = set_preditos - set_validas
                    faltando = set_validas - set_preditos
                    log(f"[AVALIAÇÃO] Classes previstas não batem.\n  → Extras: {extras}\n  → Faltando: {faltando}")
                    consenso = 0.0
                else:
                    log("[AVALIAÇÃO] Classes previstas batem exatamente com as válidas.")

            except Exception as e:
                log(f"[AVALIAÇÃO] Erro ao validar classes previstas: {e}")



        if consenso >= consenso_minimo:
            log(f"[GrampyX] Consenso alcançado ({consenso:.2f}), encerrando iteração.")
            sucesso = True
            sucesso_global = True
            
        else:
            log(f"[GrampyX] Consenso insuficiente ({consenso:.2f}), nova rodada.")
            iteracao += 1

        if not sucesso and parar_se_sucesso:
            log(f"[GrampyX] Máximo de iterações atingido para bloco {block_idx}. Partindo pro próximo.")

    log("[GrampyX] Deliberação encerrada.")
    return sucesso_global
