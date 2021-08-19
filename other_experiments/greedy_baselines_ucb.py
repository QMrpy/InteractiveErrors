"""WARNING: Incomplete Code"""

from abc import abstractmethod
import argparse
import json
import logging
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import spacy
import networkx as nx
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

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
        raise ValueError("Invalid similarity algorithm")

    return similarities


def compute_similarities(sentences, embeddings, args):        
    indexes = {}
    iteration = 0
    normalized_embeddings = []
    eps = 0.00001
    
    logging.info("Calculating node attributes")
    for sentence in tqdm(sentences):
        indexes[iteration] = sentence
        normalized_embeddings.append(embeddings[iteration] / (eps + np.linalg.norm(embeddings[iteration])))
        iteration += 1

    logging.info("Calculating edge attributes or sentence similarity scores")
    normalized_embeddings = np.array(normalized_embeddings)
    similarities_list = optimized(normalized_embeddings, args)

    similarities = {}
    for i in range(len(similarities_list)):
        for j in range(len(similarities_list)):
            similarities[(indexes[i], indexes[j])] = similarities_list[i][j]
    
    return similarities, indexes


def construct_networkx_graph(similarities, indexes, embeddings, args):
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
    graph.add_nodes_from([(v, {"node_attr": indexes[v]}) for v in range(len(embeddings))])
    graph.add_edges_from(edges)
    logging.info("Constructed graph")
    
    return graph


def get_node_ranks(graph, args):
    """ Returns nodes ranked by centralities """
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


def add_nodes(indexes, embeddings, new_embeddings, new_queries):
    """ 
    Adds new embeddings to graph. Take care of duplicates before calling this function
    Optimize this function later as this function constructs a new graph everytime called.
    Call construct_networkx_graph after this call
    """
    iteration = len(embeddings)
    normalized_embeddings = []
    eps = 0.00001
    
    logging.info("Calculating node attributes")
    for i in range(len(embeddings)):
        normalized_embeddings.append(embeddings[i] / (eps + np.linalg.norm(embeddings[i])))

    for i in range(len(new_embeddings)):
        normalized_embeddings.append(new_embeddings[i] / (eps + np.linalg.norm(new_embeddings[i])))
        indexes[iteration] = new_queries[i]
        iteration += 1

    logging.info("Calculating edge attributes or sentence similarity scores")
    normalized_embeddings = np.array(normalized_embeddings)
    similarities_list = optimized(normalized_embeddings, args)

    similarities = {}
    for i in range(len(similarities_list)):
        for j in range(len(similarities_list)):
            similarities[(indexes[i], indexes[j])] = similarities_list[i][j]
    
    return similarities, indexes


def train_UCB_algorithm(args, leakages, available_data, classification_model, embedding_model, cache):
    """Trains the UCB algorithm."""

    queries = set(leakages)
    initial_queries = set(leakages)

    if args.ucb_algorithm == "knn":
        bandit_algo = kNNUCBAlgorithm(args, available_data, initial_queries, classification_model, embedding_model, cache)
    else:
        raise ValueError("ucb_algorithm should be 'knn' ")

    iteration = 1

    while len(queries) < args.ucb_queries_generate:
        if args.ucb_algorithm == "knn" and len(initial_queries) != 0:
            query = bandit_algo.select_query(initial_queries)
            initial_queries.remove(query)

        neighbors = bandit_algo.generate_neighbors(query)

        reward = bandit_algo.compute_reward(neighbors)

        bandit_algo.update(query, reward)

        queries.remove(query)
        queries.update(neighbors)

        logging.debug(
            f"<{query}> query expanded, {len(neighbors)} neighbors produced"
            f" and got {reward:.2f} as reward. Total queries: {len(queries)}"
        )

    return bandit_algo, queries


class UCBAlgorithm:
    """Base class for UCB algorithm"""

    def __init__(self, args, initial_queries, indexes, embeddings, graph):
        """Initialise algorithm."""

        self.args = args
        self.initial_queries = initial_queries
        self.indexes = indexes
        self.embeddings = embeddings
        self.graph = graph

        logging.info(f"Loading generation model from {args.ucb_generator_model}..")
        """ Currently this script only supports T5 """
    
        try:
            logging.info("Trying to load ucb generator model if it exists..")
            self.generator_model = T5ForConditionalGeneration.from_pretrained(args.ucb_generator_model).to("cuda")
            self.generator_tokenizer = T5Tokenizer.from_pretrained(args.ucb_generator_tokenizer)
            logging.info("Generator model and tokenizer loaded.")
        except OSError:
            logging.info("Fine tuned ucb generator model not found. Aborting!!")

        logging.info("Loading spacy model..")
        self.nlp = spacy.load(args.ucb_spacy_model)
        logging.info("Spacy model loaded.")

    def random_query(self, queries):
        """Select random query."""

        return random.choice(list(queries))

    @abstractmethod
    def select_query(self, queries):
        """Select query based on the algorithm"""

        raise NotImplementedError

    def compute_reward(self, selected_query, generated_queries, graph):
        """Computes rewards based on the updated graph and generated queries.
        Generated queries should not have duplicates and should be correctly processed. """
        # compute rewards based on how much the generated queries are connected to the original graph
        # or what are the similarity scores of the queries
        # or what are the logit scores when passed through the classfier
        if len(generated_queries) == 0:
            return 0

        reward = 0                                        # calculate reward based on selected query too. Calculate rewards for each generated query
        for node in graph.nodes():
            if node >= len(self.initial_queries):
                reward += graph.neighbors(node)      

        return 0.1 * reward

    @abstractmethod
    def update(self, query, graph):
        """Updates the parameters of the algorithm"""

        raise NotImplementedError

    def generate_neighbors(self, query):
        """Generated sentences using the selected query."""

        model = (self.generator_model, self.generator_tokenizer)
        output = generate_candidate_leakages(
            model,
            [query],
            self.nlp,
            per_device_eval_batch_size=16,
            truncate_length=64,
            k=3,
            max_length=20,
            temperature=1.0,
            use_tqdm=False,
            max_generate=5,  
        )

        sentences = set([x["candidate_leakage"] for x in output])  # make distinct
        if query in sentences:
            sentences.remove(query)
        sentences = list(sentences)

        return sentences


class kNNUCBAlgorithm(UCBAlgorithm):
    """Implements i-KNN-UCB algorithm.
    https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0145-0"""

    def __init__(self):
        """Initialise algorithm."""
        super().__init__()

        self.past_rewards = {}
        for node in self.graph.nodes():
            self.past_rewards[node] = 0.1 * self.graph.neighbors(node)      # or even logit scores/simlarities can be used

    def select_query(self, queries):
        """Select query based on the algorithm """
        # compute embeddings for the queries.
        embeddings = None
        """ this dictionary maintains scores of all the queries """
        scores = {}

        """ check the paper for more information on this part """
        for query, embedding in zip(queries, embeddings):
            neighbors, distances = self._find_neighbors(nn_algo, embedding)

            expected_reward = 0
            average_distance = 0
            count = 0

            for neighbor, distance in zip(neighbors, distances):
                if distance == 0:
                    continue
                expected_reward += self.past_rewards[neighbor] / distance
                average_distance += distance
                count += 1

            expected_reward /= count
            average_distance /= count

            scores[query] = expected_reward + self.args.kucb_alpha * average_distance

        # select query with max score
        max_score = -float("inf")
        selected_query = None
        for query, score in scores.items():
            if score > max_score:
                max_score = score
                selected_query = query

        return selected_query

    def update(self, query, reward):
        """Stores reward which is to be used afterwards."""

        self.past_rewards[query] = reward

    def _find_neighbors(self, nn_algo, embedding):
        """Finds the closest sentence similarities, logits and sentences and neighbors"""

        n_neighbors = min(len(self.past_rewards), self.args.kucb_k)
        distances, indices = nn_algo.kneighbors([embedding], n_neighbors=n_neighbors)
        distances, indices = distances[0], indices[0]

        past_queries = list(self.past_rewards.keys())
        neighbors = [past_queries[index] for index in indices]
        return neighbors, distances


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
    
    logging.info("Loading data")
    data = load_data(input_file, shuffle=True)

    # load leakage data, load trained classifier, check once how many are actually leakages.

    logging.info("Computing embeddings for leakages")
    leakage_embeddings = compute_embeddings(args, leakages, trained_classification_model, convert_to_numpy=True)
    
    logging.info("Computing similarites for leakages")
    leakage_similarities, leakage_indexes = compute_similarities(leakages, leakage_embeddings, args)

    logging.info("Constructing graph for leakages")
    graph = construct_networkx_graph(leakage_similarities, leakage_indexes, None, leakage_embeddings, args)

    logging.info("Saving networkx graph and making it undirected")
    json.dump(dict(nodes=[(n, {"node_attr": graph.nodes[n]["node_attr"]}) for n in graph.nodes()],
                   edges=[[u, v] for u, v in graph.edges()]),
              open('gpt2graph.json', 'w'), indent=2)

    logging.info("Getting preferential leakages. This ranks the sentences by centrality but retains all")
    ranked_nodes = get_node_ranks(graph, args)
    ranked_leakages = [leakage_indexes[r] for r in ranked_nodes]



    '''logging.info(f"Loading generator(GPT2) model and tokenizer from {args.gen_model}..")
    generator_model = GPT2LMHeadModel.from_pretrained(args.gen_model)
    generator_model.to(device)
    generator_model.eval()

    generator_tokenizer = GPT2TokenizerFast.from_pretrained(args.gen_model)
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

    logging.info("Generating new sentences from seed leakages")
    generated_sentences_from_leakages = []
    for r in tqdm(seed_leakages):
        generated_sentences_from_leakages.extend(generate_candidate_leakages(r.strip(), generator_model, generator_tokenizer, args))    
    logging.info(f"{len(generated_sentences_from_leakages)} sentences generated.")

    with open(output_file_seed_leakage_generated_sentences, 'w') as file:
        data = {'args': vars(args), 'seed_leakage_sentences': generated_sentences_from_leakages}
        json.dump(data, file, indent=2)
    logging.info(f"Generated sentences from seeds written to {output_file_seed_leakage_generated_sentences}.")
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".", help="Directory with input data")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory where to save output")
    parser.add_argument("--input_data", type=str, help="Input file containing sentences in [score, query] format")
    parser.add_argument("--output_file_seed_generated", type=str, default="GPT2_generated.json", help="Output file with generated sentences")
    parser.add_argument("--output_file_generated_from_seed_leakages", type=str, default="GPT2_generated_sentences_from_seed_leakages.json", help="Output file with generated sentences from seed leakages")
    parser.add_argument("--gen_model", type=str, default="gpt2-medium")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to take from a large input dataset")
    parser.add_argument("--min_prefix_length", type=int, default=2)
    parser.add_argument("--max_prefix_length", type=int, default=5)
    parser.add_argument("--similarity_algo", type=str, default="euclidean", help="Similarity algorithm to find similarities between sentences")
    parser.add_argument("--sim_threshold", type=float, default=0.7, help="Maximum Euclidean distance between embeddings for an edge to exist between corresponding sentence nodes")
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