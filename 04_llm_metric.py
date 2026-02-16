import os
import json
import numpy as np
import torch
from tqdm import trange
import information_geometry as ig
import argparse
import torch.nn.functional as F
from collections import defaultdict

MODEL_NAME = "google/gemma-3-4b-pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

model, tokenizer, vocab_dict, vocab_list, G = ig.load_model_and_vocab(MODEL_NAME, device=DEVICE)

arg = argparse.ArgumentParser()
arg.add_argument("--concept_name", type=str, default="verb_en_fr")
args = arg.parse_args()

concept_name = args.concept_name


base_path = "LLM_BASE_PATH" # Replace with the actual base path where data is stored
concept_path = os.path.join(base_path, concept_name)
primals_dict_list = torch.load(os.path.join(concept_path, "test_steering_paths.pt"))

with open(os.path.join(f"data/mapping_{concept_name}.json"), "r") as f:
    mapping = json.load(f)
mapping = ig.get_clean_mapping(mapping, vocab_dict)

directions = torch.load(os.path.join(concept_path, "directions.pt"))

indices0 = [vocab_dict[i] for i in list(set(mapping.keys()))]
indices1 = [vocab_dict[i] for i in list(set(mapping.values()))]


all_list = {dir: defaultdict(list) for dir in directions.keys()}

for dir_name in directions.keys():
    for type in ['e', 'm']:
        for i in trange(len(primals_dict_list)):
            path = primals_dict_list[i][dir_name][type]

            probs = path @ G.T
            probs = F.softmax(probs, dim=-1)

            probs0, probs1 = ig.get_base_target_probs(probs, indices0, indices1)
            cf_sum = probs0 + probs1
            ratio = probs1 / (probs0 + probs1 + 1e-10)
            mask = ratio < 0.9999

            off_prob = ig.get_off_probs(probs, mapping, vocab_dict)
            fkl = ig.get_kls(off_prob, offset = 1e-6)
            rank_diff = ig.get_rank_diff(off_prob, use_topp = True)
            cos = ig.get_cos(probs, G, directions[dir_name])

            all_list[dir_name][type].append({
                "probs0": probs0[mask].cpu(),
                "probs1": probs1[mask].cpu(),
                "sum": cf_sum[mask].cpu(),
                "ratio": ratio[mask].cpu(),
                "fkl": fkl[mask].cpu(),
                "rank_diff": rank_diff[mask].cpu(),
                "cos": cos[mask].cpu()
            })

torch.save(all_list, os.path.join(concept_path, "test_steering_metrics.pt"))
