# Dual-Steering
This repository contains the official implementation of Dual Steering, the practical application proposed in [our paper](https://www.arxiv.org/abs/2602.15293).

This codebase implements Euclidean and Dual steering for the following models: Gemma-3-4B (LLM representations) and MetaCLIP-2 (Vision-Language representations). It also includes the code for computing the metrics reported in the paper to validate the method's effectiveness.

## Data
* **Counterfactual Pairs**: Each `mapping_verb_***.json` file contains a set of counterfactual token pairs corresponding to specific binary concepts.
* **Context Embeddings**: The training and test sets for context embeddings are generated using `01_llm_data.ipynb` and `02_llm_embeddings.py`. All contexts for LLM experiments are sampled from the C4 dataset.
* **Synthetic Object Image Dataset**: The dataset is provided as a zip file on Google Drive: [Download the dataset](https://drive.google.com/file/d/1I8wjiU4eTxkmlVHFZd9-QQZdAVXFM9Av/view?usp=sharing).

## Experiments
* **`03_llm_steering.py`**: Performs steering on context embeddings within the LLM test sets, applying both Euclidean and Dual methods.
* **`04_llm_metric.py`**: Computes various metrics to evaluate each steering path.
* **`05_clip_object.py`**: Implements the steering process and evaluates metrics for the synthetic object datasets using the CLIP model.
* **`06_clip_coco.py`**: Implements the steering process and evaluates metrics for the COCO datasets using the CLIP model.
* **`07_visualization.ipynb`**: Visualizes metrics of all experiments.