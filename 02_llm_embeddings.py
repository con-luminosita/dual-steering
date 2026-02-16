import os
import json
import information_geometry as ig
import argparse

arg = argparse.ArgumentParser()
arg.add_argument('--model_name', type=str, default='google/gemma-3-4b-pt', help='LLM model name')
arg.add_argument('lang', type=str, help='Language code (e.g., en, fr)')
arg.add_argument('shard_idx', type=int, help='Shard index to process')
args = arg.parse_args()


lang = args.lang
shard_idx = args.shard_idx

base_path = "LLM_BASE_PATH" # Replace with the actual base path where data is stored
save_path = os.path.join(base_path, f'embeddings/{lang}_embeddings_shard_{shard_idx}.pt')

with open(os.path.join(base_path, f'embeddings/{lang}_texts_shard_{shard_idx}.json'), 'r') as f:
    texts = json.load(f)


ig.get_llm_embeddings(
    texts,
    args.model_name,
    save_to=save_path,
    batch_size=16,
    last_position_only=False,
    max_length=256
)