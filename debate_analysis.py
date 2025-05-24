import json
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_debate_logs(evaluation_logs, save_path="debate_victories.png"):
    win_count = defaultdict(int)
    agreement_count = defaultdict(int)
    isolation_count = defaultdict(int)
    participation_count = defaultdict(int)

    for result in evaluation_logs.values():
        for round_data in result.get("history", []):
            candidates = round_data.get("candidates", [])
            winner_idx = round_data.get("winner")

            valid_responses = [c for c in candidates if c is not None]
            if not valid_responses:
                continue

            vote_counts = defaultdict(int)
            for c in valid_responses:
                vote_counts[json.dumps(c)] += 1

            if winner_idx is not None:
                win_count[f"model_{winner_idx}"] += 1
                winner_response = candidates[winner_idx]
            else:
                winner_response = None

            for i, response in enumerate(candidates):
                model_key = f"model_{i}"
                if response is not None:
                    participation_count[model_key] += 1

                    if response == winner_response and i != winner_idx:
                        agreement_count[model_key] += 1

                    count = vote_counts.get(json.dumps(response), 0)
                    if count == 1:
                        isolation_count[model_key] += 1

    models = sorted(participation_count.keys())
    print("\n=== DEBATE METRICS ===")
    for model in models:
        p = participation_count[model]
        w = win_count[model]
        a = agreement_count[model]
        iso = isolation_count[model]
        win_rate = w / p * 100 if p else 0
        agree_rate = a / p * 100 if p else 0
        iso_rate = iso / p * 100 if p else 0
        print(f"\n{model}")
        print(f" Participações: {p}")
        print(f" Vitórias: {w} ({win_rate:.1f}%)")
        print(f" Concordâncias com maioria: {a} ({agree_rate:.1f}%)")
        print(f" Respostas isoladas: {iso} ({iso_rate:.1f}%)")

    plt.figure(figsize=(10, 5))
    bars = [win_count[m] for m in models]
    plt.bar(models, bars, color='darkcyan')
    plt.title("Número de vitórias por modelo")
    plt.ylabel("Vitórias")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()