"""
This file takes in training data, and finds classifier labels by training or loading
a classification model (roberta-base). It constructs a graph out of the seed
data, which are some randomly sampled sentences from the trainset. It ranks the
nodes by centrality scores, retains all of them, and generates sentences from 
the ranked sentences using GPT2. The generated graph can be plotted to get the leakages.
"""

import argparse
import json
import logging
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import networkx as nx
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import load_data, split_data, separate_labels
from classification import compute_embeddings, train_classification_model, find_leakages
from gpt2 import generate_candidate_leakages


def seed(RANDOM_SEED=42):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)


def optimized(embeddings, args):
    """ Computes the similarity calculations in an optimized manner. Later CUDA or Numba maybe used to parallelize """
    similarities = np.zeros(shape=(len(embeddings), len(embeddings)))
    
    if args.similarity_algo == "cosine":
        for i in tqdm(range(len(embeddings))):
            for j in range(len(embeddings)):
                similarities[i][j] = np.dot(embeddings[i], np.transpose(embeddings[j]))

    else:
        raise NotImplementedError("Invalid similarity algorithm")

    return similarities


def compute_similarities(sentences, embeddings, args):        
    indexes = {}
    iteration = 0
    
    logging.info("Calculating node attributes")
    for sentence in tqdm(sentences):
        indexes[iteration] = sentence
        iteration += 1

    logging.info("Calculating edge attributes or sentence similarity scores")
    similarities_list = optimized(embeddings, args)

    similarities = {}
    for i in range(len(similarities_list)):
        for j in range(len(similarities_list)):
            similarities[(indexes[i], indexes[j])] = similarities_list[i][j]
    
    return similarities, indexes


def construct_networkx_graph(similarities, indexes, labels, embeddings, args):
    edges = []

    if args.similarity_algo == "cosine":
        for i in tqdm(indexes):
            for j in indexes:
                if similarities[(indexes[i], indexes[j])] >= args.sim_threshold :                       
                    edges.append((i, j))
                    if i != j:
                        edges.append((j, i))
    else:
        raise ValueError("Invalid similarity algorithm")
                    
    graph = nx.Graph()
    embeddings = np.squeeze(embeddings)
    graph.add_nodes_from([(v, {"node_attr": indexes[v], "labels": labels[v]}) for v in range(len(embeddings))])
    graph.add_edges_from(edges)
    logging.info("Constructed graph")
    
    return graph


def get_neighbors_and_rank(graph, args):
    """ Returns immediate neighbors of nodes ranked by degrees """

    ranked_nodes = []
    for node in graph.nodes():
        ranked_nodes.append(node)

    centrality = []
    if args.centrality_algo == "eigenvector":
        centrality = nx.eigenvector_centrality(graph)
    elif args.centrality_algo == "degree":
        centrality = nx.degree_centrality(graph)
    else:
        raise NotImplementedError("Invalid centrality algorithm")

    centrality = [centrality[i] for i in graph.nodes()]    
    ranked_nodes = sorted(ranked_nodes, key=lambda x: centrality[x], reverse=True)
    
    return ranked_nodes


def seed(RANDOM_SEED=42):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)


def main(args):
    RANDOM_SEED = 42
    device = "cuda"

    seed(RANDOM_SEED)

    input_file = os.path.join(args.input_dir, args.input_data)
    output_file_generated_sentences = os.path.join(args.output_dir, args.output_file_seed_generated)

    logging.info("Loading data and sampling")
    data = load_data(input_file, shuffle=True)
    sampled_indices = np.random.choice(np.arange(len(data)), size=args.samples, replace=False)
    sampled_data = [data[i] for i in sampled_indices]                                      # shuffle data before choosing else no sentence with label 1 will be sampled, as they lie far apart

    logging.info("Splitting data into train, validation and seed datasets")
    train_data, val_data, seed_data = split_data(sampled_data, [0.6, 0.38, 0.02], seed=RANDOM_SEED)
    seed_sentences, _ = separate_labels(seed_data)
    logging.info(f"Created train ({len(train_data)} examples), valid ({len(val_data)} examples) and seed ({len(seed_data)} examples) dataset.")

    logging.info("Fine tuning classification model")
    trained_classification_model = train_classification_model(args, train_data, val_data)

    logging.info("Finding leakages for seed data")
    _, plot_labels = find_leakages(args, trained_classification_model, seed_data)

    logging.info("Computing embeddings for seed sentences")
    seed_embeddings = compute_embeddings(args, seed_sentences, trained_classification_model, convert_to_numpy=True)
    
    logging.info("Computing similarites for seed sentences")
    seed_similarities, seed_indexes = compute_similarities(seed_sentences, seed_embeddings, args)

    logging.info("Constructing graph for seed sentences")
    graph = construct_networkx_graph(seed_similarities, seed_indexes, plot_labels, seed_embeddings, args)

    logging.info("Saving networkx graph and making it undirected")
    json.dump(dict(nodes=[(n, {"node_attr": graph.nodes[n]["node_attr"], "labels": graph.nodes[n]["labels"]}) for n in graph.nodes()],
                    edges=[[u, v] for u, v in graph.edges()]),
                    open('gpt2graph.json', 'w'), indent=2)

    logging.info("Getting preferential seed sentences. This ranks the sentences by centrality but retains all")
    ranked_seed_nodes = get_neighbors_and_rank(graph, args)
    ranked_seed_sentences = [seed_indexes[r] for r in ranked_seed_nodes]

    logging.info(f"Loading generator(GPT2) model and tokenizer from {args.gen_model}..")
    generator_model = GPT2LMHeadModel.from_pretrained(args.gen_model)
    generator_model.to(device)
    generator_model.eval()

    generator_tokenizer = GPT2Tokenizer.from_pretrained(args.gen_model)
    generator_tokenizer.pad_token = generator_tokenizer.eos_token
    logging.info("Generator model and tokenizer loaded.")

    logging.info("Generating new sentences from ranked seed sentences")
    generated_sentences = []
    for r in tqdm(ranked_seed_sentences):
        generated_sentences.extend(generate_candidate_leakages(r.strip(), generator_model, generator_tokenizer, args))    
    logging.info(f"{len(generated_sentences)} sentences generated.")

    with open(output_file_generated_sentences, 'w') as file:
        data = {'args': vars(args), 'seed_sentences': generated_sentences}
        json.dump(data, file, indent=2)
    logging.info(f"Generated sentences from seeds written to {output_file_generated_sentences}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".", help="Directory with input data")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory where to save output")
    parser.add_argument("--input_data", type=str, help="Input file containing sentences in [score, query] format")
    parser.add_argument("--output_file_seed_generated", type=str, default="GPT2_generated.json", help="Output file with generated sentences")
    parser.add_argument("--gen_model", type=str, default="gpt2-medium")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to take from a large input dataset")
    parser.add_argument("--min_prefix_length", type=int, default=2)
    parser.add_argument("--max_prefix_length", type=int, default=5)
    parser.add_argument("--similarity_algo", type=str, default="cosine", help="Similarity algorithm to find similarities between sentences")
    parser.add_argument("--sim_threshold", type=float, default=0.5, help="Maximum Euclidean distance between embeddings for an edge to exist between corresponding sentence nodes")
    parser.add_argument("--centrality_algo", type=str, default="eigenvector", help="Centrality algorithm to rank nodes in graph")
    parser.add_argument("--exp_name", type=str, default=str(time.time()))
    parser.add_argument("--cls_model", type=str, default="roberta-base")
    parser.add_argument("--cls_model_save_name", type=str, default="roberta-base")
    parser.add_argument("--cls_overwrite_output_dir", action="store_true")
    parser.add_argument("--cls_per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--cls_per_device_eval_batch_size", type=int, default=4096)
    parser.add_argument("--cls_learning_rate", type=float, default=1e-5)
    parser.add_argument("--cls_num_train_epochs", type=int, default=3)
    parser.add_argument("--cls_max_seq_length", type=int, default=40)
    parser.add_argument("--cls_model_save_name", type=str, default="classification_model")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()

    logging_level = logging.INFO if not args.quiet else logging.ERROR
    logging.basicConfig(level=logging_level)

    main(args)