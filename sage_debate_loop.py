import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import tensorflow as tf
from runtime_utils import log, pad_to_shape
from collections import defaultdict, Counter

WINNING_VOTES_COUNT = 3

def conversational_loop(models, input_grid, max_rounds=100):
    def generate_response(model):
        x = tf.convert_to_tensor([pad_to_shape(tf.convert_to_tensor(input_grid, dtype=tf.int32))])
        x_onehot = tf.one_hot(x, depth=15, dtype=tf.float32)
        y_pred = model(x_onehot, training=False)
        return tf.argmax(y_pred[0], axis=-1).numpy().tolist()

    log("[INFO] Iniciando debate com múltiplas rodadas")
    log(json.dumps(input_grid))

    all_responses = []
    model_output_counter = Counter()
    model_similarity_scores = defaultdict(list)

    round_num = 1
    while round_num <= max_rounds:
        log(f"[INFO] Rodada {round_num} iniciada")
        responses = []
        for i, model in enumerate(models):
            try:
                output = generate_response(model)
                model_output_counter[f"model_{i+1}"] += 1
                responses.append(output)
            except Exception:
                responses.append(None)

        valid_responses = [r for r in responses if r is not None]
        round_entry = {
            "candidates": responses,
            "votes": [],
            "winner": None
        }

        def count_votes(candidates):
            votes = defaultdict(int)
            for c in candidates:
                key = json.dumps(c)
                votes[key] += 1
            most_common = max(votes.items(), key=lambda x: x[1])
            return json.loads(most_common[0]), most_common[1], votes

        if valid_responses:
            voted_output, count, all_votes = count_votes(valid_responses)
            round_entry["votes"] = [json.dumps(r) for r in valid_responses]

            if count >= WINNING_VOTES_COUNT:
                winner_idx = responses.index(voted_output)
                round_entry["winner"] = winner_idx
                all_responses.append(round_entry)
                log(f"[INFO] Votação encerrada com maioria na rodada {round_num}")
                return {
                    "output": voted_output,
                    "success": True,
                    "rounds": round_num,
                    "history": all_responses,
                    "output_diversity": dict(model_output_counter),
                    "similarity_scores": {k: sum(v)/len(v) for k,v in model_similarity_scores.items() if v}
                }

            for i, output in enumerate(responses):
                if output is not None:
                    similarity = sum([int(output == other) for other in responses if other is not None and other != output])
                    model_similarity_scores[f"model_{i+1}"].append(similarity)

        all_responses.append(round_entry)
        round_num += 1

    # Fallback: retorna o mais votado mesmo sem maioria
    fallback_output, _, _ = count_votes([r for r in responses if r is not None])
    log("[INFO] Debate finalizado sem maioria. Retornando resultado mais votado por fallback.")
    return {
        "output": fallback_output,
        "success": False,
        "rounds": max_rounds,
        "history": all_responses,
        "output_diversity": dict(model_output_counter),
        "similarity_scores": {k: sum(v)/len(v) for k,v in model_similarity_scores.items() if v}
    }