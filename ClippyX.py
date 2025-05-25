# ClippyX_persistente.py
import tensorflow as tf
import numpy as np
import os
import pickle
from court_logic import arc_court_supreme
from confidence_system import ConfidenceManager
from metrics_utils import plot_prediction_test, gerar_video_time_lapse, embutir_trilha_sonora
from runtime_utils import save_debug_result
from metrics_utils import log
from models_loader import load_model
from data_pipeline import load_data_batches
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from ClippyX_internal_models import ClippyInternalModels

PERSIST_DIR = "clippy_data"
DETECTOR_WEIGHTS = os.path.join(PERSIST_DIR, "detector.h5")
HISTORY_PATH = os.path.join(PERSIST_DIR, "history.pkl")

class ClippyX:
    def __init__(self, num_modelos=5):
        os.makedirs(PERSIST_DIR, exist_ok=True)

        self.num_modelos = num_modelos
        self.models = [load_model(i) for i in range(num_modelos)]
        self.manager = ConfidenceManager(self.models)
        self.submission_dict = []
        self.history_X = []
        self.history_y = []
        self.internal_models = ClippyInternalModels()

        self.detector = Sequential([
            Input(shape=(30 * 30,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.detector.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        if os.path.exists(DETECTOR_WEIGHTS):
            self.detector.load_weights(DETECTOR_WEIGHTS)
            log("[CLIPPYX] Pesos do detector carregados")

        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "rb") as f:
                self.history_X, self.history_y = pickle.load(f)
            log("[CLIPPYX] Histórico carregado")

    def salvar_estado(self):
        self.detector.save_weights(DETECTOR_WEIGHTS)
        with open(HISTORY_PATH, "wb") as f:
            pickle.dump((self.history_X, self.history_y), f)
        log("[CLIPPYX] Estado salvo")

    def preparar_inputs(self, x):
        if x.shape[-1] == 40:
            x_juiz = x
            x_outros = tf.concat(tf.split(x, num_or_size_splits=10, axis=-1)[:1], axis=-1)
        else:
            x_outros = x
            x_juiz = tf.zeros((1, 30, 30, 1, 40), dtype=tf.float32)
        return x_outros, x_juiz

    def julgar(self, x_input, raw_input, block_index, task_id):
        try:
            x_input = tf.expand_dims(tf.convert_to_tensor(x_input, dtype=tf.float32), axis=0)
            x_outros, _ = self.preparar_inputs(x_input)

            if len(self.models) < 6:
                self.models.append(load_model(5))  # Suprema Juíza

            log(f"[CLIPPYX] Julgando bloco {block_index} — Task {task_id}")
            resultados = arc_court_supreme(
                self.models,
                x_outros,
                task_id=task_id,
                block_idx=block_index,
                confidence_manager=self.manager
            )

            consenso = resultados.get("consenso", 0.0)
            y_pred = resultados["class_logits"] if isinstance(resultados, dict) else resultados
            y_pred_np = y_pred.numpy() if isinstance(y_pred, tf.Tensor) else y_pred

            flat = y_pred_np.flatten()
            label = 1.0 if consenso >= 0.9 else 0.0
            self.history_X.append(flat)
            self.history_y.append(label)

            self.internal_models.adicionar_voto(y_pred_np, consenso)
            self.internal_models.treinar_todos()

            if len(self.history_X) >= 10:
                self.detector.fit(np.array(self.history_X), np.array(self.history_y), epochs=5, verbose=0)
                log(f"[CLIPPYX] Detector interno treinado")

            self.salvar_estado()

            salvar_path = f"images/clippy/JULGAMENTO_{block_index}_{task_id}"
            plot_prediction_test(raw_input=raw_input, predicted_output=y_pred_np, save_path=salvar_path)

            self.submission_dict.append({"task_id": task_id, "prediction": y_pred_np})
            save_debug_result(self.submission_dict, "submission.json")

            video_path = gerar_video_time_lapse("votos_visuais", block_index, f"{block_index}_{task_id}.avi")
            if video_path:
                embutir_trilha_sonora(video_path, block_index)

            return {"consenso": consenso}

        except Exception as e:
            log(f"[CLIPPYX ERRO] Bloco {block_index}: {str(e)}")
            return {"consenso": 0.0}


def rodar_deliberacao_com_condicoes(parar_se_sucesso=True, max_iteracoes=10, consenso_minimo=0.9):
    clippy = ClippyX()
    batches = load_data_batches()

    for (X, raw_input, block_idx, task_id) in batches:
        iteracao = 0
        sucesso = False

        while not sucesso and iteracao < max_iteracoes:
            print(f"[CLIPPYX] Deliberação iter {iteracao} — Task {task_id} — Bloco {block_idx}")
            resultado = clippy.julgar(X, raw_input, block_idx, task_id)

            consenso = resultado.get("consenso", 0)
            if consenso >= consenso_minimo:
                print(f"[CLIPPYX] Consenso alcançado ({consenso:.2f}), encerrando iteração.")
                sucesso = True
            else:
                print(f"[CLIPPYX] Consenso insuficiente ({consenso:.2f}), nova rodada.")
                iteracao += 1

        if not sucesso and parar_se_sucesso:
            print(f"[CLIPPYX] Máximo de iterações atingido para bloco {block_idx}. Partindo pro próximo.")

    print("[CLIPPYX] Deliberação encerrada.")
    return False


if __name__ == "__main__":
    while True:
        while rodar_deliberacao_com_condicoes(
            parar_se_sucesso=True,
            max_iteracoes=150,
            consenso_minimo=0.9
        ):
            pass
