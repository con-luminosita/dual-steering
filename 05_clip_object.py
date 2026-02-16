import information_geometry as ig

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from tqdm import trange
import argparse
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


from transformers import AutoProcessor, MetaClip2Model

MODEL_NAME = "facebook/metaclip-2-worldwide-huge-quickgelu"
clip_processor = AutoProcessor.from_pretrained(MODEL_NAME)
clip_model = MetaClip2Model.from_pretrained(MODEL_NAME).to(DEVICE).eval()



##### Setup datasets and vocabularies #####
# Path to the downloaded synthetic object image dataset
data_path = "/path/to/synthetic_object_dataset"  # Replace this with your local path
concept_dict = {
    "shape": ["circles", "squares", "triangles"],
    "color": ["red", "green", "blue", "yellow"],
    "number": [4],
}

G, image_dataset, combo_dataset = ig.load_color_shape_image(
    data_path, concept_dict, clip_model, clip_processor,
    image_num_images = 2, combo_num_images = 2,
)

original_vocab_list = [f"{sample['color']}_{sample['shape']}" for sample in image_dataset.samples]
original_vocab_list += [f"{sample['color1']}_{sample['shape1']}_{sample['color2']}_{sample['shape2']}" for sample in combo_dataset.samples]

from datasets import load_dataset
ds = load_dataset("detection-datasets/coco", split="val")

# COCO indices by category:
sampled_indices = {
    # cat_to_dog
    'bicycle': [1757, 1930],
    'cat': [800, 551],
    'cat + bicycle': [857, 913],
    'cat + dog': [219, 1893],
    'dog': [271, 1164],
    'dog + bicycle': [3621, 1172],
    # carrot_to_broccoli
    'broccoli': [147, 2867],
    'broccoli + pizza': [788, 4936],
    'carrot': [1208, 1707],
    'carrot + broccoli': [1331, 2700],
    'carrot + pizza': [3714, 3897],
    'pizza': [235, 1353],
    # traffic_light_to_fire_hydrant
    'bus': [48, 890],
    'fire hydrant': [1642, 1165],
    'fire hydrant + bus': [888, 1362],
    'traffic light': [745, 802],
    'traffic light + bus': [1404, 3878],
    'traffic light + fire hydrant': [4842, 2891],
}

coco_labels = []
indices = []
for k, v in sampled_indices.items():
    for idx in v:
        coco_labels.append(k)
        indices.append(idx)
images = ds[indices]["image"]
coco_G = ig.get_clip_image_embeddings(images, clip_model, clip_processor)


G = torch.cat([G, coco_G], dim = 0)
original_vocab_list += coco_labels
vocab_list = sorted(list(set(original_vocab_list)))
vocab_dict = {vocab: i for i, vocab in enumerate(vocab_list)}

print(f"Original vocab size: {len(original_vocab_list)}")
print(f"Unique vocab size: {len(vocab_list)}")



arg = argparse.ArgumentParser()
arg.add_argument("--target_concept", type=str, default="color")
arg.add_argument("--other_concept", type=str, default="shape")
arg.add_argument("--value0", type=str, default="blue")
arg.add_argument("--value1", type=str, default="red")
args = arg.parse_args()

target_concept = args.target_concept
other_concept = args.other_concept
value0 = args.value0
value1 = args.value1


#### counterfactual mapping #####
print(f"Target concept: {target_concept}, Other concept: {other_concept}, Value0: {value0}, Value1: {value1}")

mapping = {}
for i, vocab in enumerate(vocab_list):
    if value0 in vocab:
        if value1 in vocab:
                continue
        if len(vocab.split('_')) <= 2:
            target_word = vocab.replace(value0, value1)
            mapping[vocab] = target_word
        else:
            target_word = vocab.replace(value0, value1)
            if target_word not in vocab_list:
                if target_concept == "color":
                    parts = vocab.split('_')
                    if parts[0] == value0:
                        parts[0] = value1
                    elif parts[2] == value0:
                        parts[2] = value1
                    target_word = '_'.join(parts)
                else:
                    parts = vocab.split('_')
                    if parts[1] == value0:
                        parts[1] = value1
                    elif parts[3] == value0:
                        parts[3] = value1
                    target_word = '_'.join(parts)
            if target_word in vocab_list:
                mapping[vocab] = target_word

print(f"Number of mapping: {len(mapping)}")




#### Prepare train and test sets #####
prefix_formats = [
    "",
    "a rendering of ",
    "a depiction of ",
    "an illustration of ",
    "a conceptual illustration of ",
    "A rendering of ",
    "A depiction of ",
    "An illustration of ",
    "A conceptual illustration of ",
    "Rendering of ",
    "Depiction of ",
    "Illustration of ",
    "Conceptual illustration of ",
]

def get_train_test(concept_dict, value0, value1, other_concept, prefix_formats, alpha = 0.7):
    single_texts0 = []
    single_texts1 = []
    for pf in prefix_formats:
        if other_concept == "shape":
            for s in concept_dict[other_concept]:
                single_texts0.append(f"{pf}{value0} {s}")
                single_texts1.append(f"{pf}{value1} {s}")
        elif other_concept == "color":
            for c in concept_dict[other_concept]:
                single_texts0.append(f"{pf}{c} {value0}")
                single_texts1.append(f"{pf}{c} {value1}")

    indices = list(range(len(single_texts0)))
    random.seed(100)
    random.shuffle(indices)
    split_idx = int(alpha * len(indices))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_texts0 = [single_texts0[i] for i in train_indices]
    test_texts0 = [single_texts0[i] for i in test_indices]
    train_texts1 = [single_texts1[i] for i in train_indices]
    test_texts1 = [single_texts1[i] for i in test_indices]

    train_primals0 = ig.get_clip_text_embeddings(train_texts0, clip_model, clip_processor)
    train_primals1 = ig.get_clip_text_embeddings(train_texts1, clip_model, clip_processor)
    test_primals0 = ig.get_clip_text_embeddings(test_texts0, clip_model, clip_processor)
    test_primals1 = ig.get_clip_text_embeddings(test_texts1, clip_model, clip_processor)

    return test_texts0, train_primals0, train_primals1, test_primals0, test_primals1


def aggregate_probs(probs, original_vocab_list, vocab_list):
    new_probs = torch.zeros(probs.size(0), len(vocab_list)).to(probs.device)
    for i, vocab in enumerate(vocab_list):
        for j, original_vocab in enumerate(original_vocab_list):
            if vocab == original_vocab:
                new_probs[:, i] += probs[:, j]
    return new_probs


test_texts0, train_primals0, train_primals1, test_primals0, test_primals1 = get_train_test(concept_dict, value0, value1, other_concept, prefix_formats, alpha = 0.7)
train_duals0 = ig.primals_to_duals(train_primals0, G)
train_duals1 = ig.primals_to_duals(train_primals1, G)





#### Compute directions #####
directions = {
    "primal_md": ig.get_MD(train_primals0, train_primals1),
    "dual_md": ig.get_MD(train_duals0, train_duals1),
}
direction_names = list(directions.keys())



#### Steering test primals #####
primals_dict_list = []
for start_primal in tqdm(test_primals0):
    paths = {name: {'e': None, 'm': None} for name in direction_names}
    for name in direction_names:
        direction = directions[name]
        e_path = ig.e_steering(start_primal, direction, G, num_steps = 400, step_size = 0.1,  use_tqdm=False)
        m_path = ig.m_steering(start_primal, direction, G, alpha = 5e-3, num_steps = 3000, step_size = 0.5, use_tqdm=False)
        paths[name]['e'] = e_path
        paths[name]['m'] = m_path
    primals_dict_list.append(paths)





#### Compute metrics along the paths #####
indices0 = [vocab_dict[i] for i in list(set(mapping.keys()))]
indices1 = [vocab_dict[i] for i in list(set(mapping.values()))]

all_list = {dir: defaultdict(list) for dir in directions.keys()}

for dir_name in directions.keys():
    for method in ['e', 'm']:
        for i in trange(len(primals_dict_list)):
            path = primals_dict_list[i][dir_name][method]

            probs = path @ G.T
            probs = F.softmax(probs, dim=-1)
            cos = ig.get_cos(probs, G, directions[dir_name])


            ### For CLIP ###
            probs = aggregate_probs(probs, original_vocab_list, vocab_list)

            probs0, probs1 = ig.get_base_target_probs(probs, indices0, indices1)
            cf_sum = probs0 + probs1
            ratio = probs1 / (probs0 + probs1 + 1e-10)
            mask = ratio < 0.9999


            off_prob = ig.get_off_probs(probs, mapping, vocab_dict)
            fkl = ig.get_kls(off_prob, offset = 1e-6)
            rank_diff = ig.get_rank_diff(off_prob, use_topp = False)
            

            all_list[dir_name][method].append({
                "probs0": probs0[mask].cpu(),
                "probs1": probs1[mask].cpu(),
                "sum": cf_sum[mask].cpu(),
                "ratio": ratio[mask].cpu(),
                "fkl": fkl[mask].cpu(),
                "rank_diff": rank_diff[mask].cpu(),
                "cos": cos[mask].cpu()
            })

base_path = "CLIP_BASE_PATH" # Replace with the actual base path where data is stored
torch.save(all_list, os.path.join(base_path, f"{target_concept}_{value0}_to_{value1}.pt"))