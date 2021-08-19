"""WARNING: Incomplete code for Thompson Sampling"""

import argparse
import logging
import os
import pickle
import random
import time
from statistics import mean

import numpy as np
import spacy
import pymc3 as pm
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import NearestNeighbors
from transformers import T5TokenizerFast, T5ForConditionalGeneration


from classification import (
    compute_embeddings,
    find_leakages,
    get_embedding_model,
    run_classification_model,
    train_classification_model,
)

from density_estimate import get_batch_density_estimates
from generator.t5 import generate_candidate_leakages
from generator.utils import write_file
from utils import load_data, split_data, separate_labels


def main(args):
    """Main control function."""

    cache_fp = os.path.join(args.output_dir, args.cache_file)
    if not args.save_cache and os.path.isfile(cache_fp):
        logging.info(f"Loading cache from {cache_fp}..")
        with open(cache_fp, "rb") as file:
            cache = pickle.load(file)
        logging.info("Cache loaded.")
        if "density_estimate" not in cache:
            cache["density_estimate"] = {}
    else:
        logging.info("Initialising empty cache..")
        cache = {"prediction_map": {}, "embedding_map": {}, "density_estimate": {}}

    data_fp = os.path.join(args.data_dir, args.labelled_data)
    logging.info(f"Reading data from {data_fp}..")
    data = load_data(data_fp)
    if args.debug:
        data = random.sample(data, 1000)
    logging.info(f"Read {len(data)} examples.")

    train_data, val_data, prod_data = split_data(data, ratio=args.data_split_ratio, seed=args.seed)

    classification_model = train_classification_model(args, train_data, val_data)

    texts, _ = separate_labels(data)

    if len(cache["prediction_map"]) == 0:
        logging.info("Running classification model on all the queries..")
        predictions, _ = run_classification_model(args, classification_model, texts, progress_bar=True)
        for text, prediction in zip(texts, predictions):
            cache["prediction_map"][text] = prediction
        logging.info(f"Obtained predictions for {len(cache['prediction_map'])} queries.")

    embedding_model = {
        "model": get_embedding_model(classification_model["model"]),
        "tokenizer": classification_model["tokenizer"],
    }

    if len(cache["embedding_map"]) == 0:
        logging.info("Computing embeddings for all the queries..")
        embeddings = compute_embeddings(args, texts, embedding_model, convert_to_numpy=True, progress_bar=True)
        for text, embedding in zip(texts, embeddings):
            cache["embedding_map"][text] = embedding
        logging.info(f"Computed embeddings for {len(cache['embedding_map'])} queries.")

    logging.info("Finding leakages..")
    leakages = find_leakages(classification_model, prod_data, cache)
    logging.info(f"Found {len(leakages)} leakages.")

    available_data = train_data + val_data
    if args.debug:
        leakages = random.sample(leakages, 5)
    ucb_algo, queries = train_UCB_algorithm(
        args, leakages, available_data, classification_model, embedding_model, cache
    )

    queries_save_path = os.path.join(args.output_dir, args.ucb_queries_fp)
    write_file(list(queries), queries_save_path)

    if args.save_cache:
        logging.info(f"Saving cache to {cache_fp}..")
        with open(cache_fp, "wb") as file:
            pickle.dump(cache, file)
        logging.info("Cache saved.")


def train_UCB_algorithm(args, leakages, available_data, classification_model, embedding_model, cache):
    """Trains the UCB algorithm."""

    log_dir = os.path.join(args.output_dir, "ucb_logs")
    writer = SummaryWriter(log_dir=log_dir)

    queries = set(leakages)
    initial_queries = set(leakages)

    if args.ucb_algorithm == "knn":
        bandit_algo = kNNUCBAlgorithm(args, available_data, initial_queries, classification_model, embedding_model, cache)
    elif args.ucb_algorithm == "lin":
        bandit_algo = LinUCBAlgorithm(args, available_data, initial_queries, classification_model, embedding_model, cache)
    elif args.ucb_algorithm == "thompson":
        bandit_algo = ThompsonSampling(args, available_data, initial_queries, classification_model, embedding_model, cache)
    else:
        raise ValueError("ucb_algorithm could be either 'knn' or 'lin' or 'thompson'.")

    iteration = 1

    while len(queries) < args.ucb_queries_generate:
        if args.ucb_algorithm == "knn" and len(initial_queries) != 0:
            query = bandit_algo.random_query(initial_queries)
            initial_queries.remove(query)
        elif args.ucb_algorithm == "thompson":
            query = bandit_algo.select_query()                 
        else:
            query = bandit_algo.select_query(queries)

        neighbors = bandit_algo.generate_neighbors(query)

        reward = bandit_algo.compute_reward(neighbors)

        bandit_algo.update(query, reward)

        queries.remove(query)
        queries.update(neighbors)

        logging.debug(
            f"<{query}> query expanded, {len(neighbors)} neighbors produced"
            f" and got {reward:.2f} as reward. Total queries: {len(queries)}"
        )

        writer.add_scalar("reward", reward, iteration)
        writer.add_scalar("num_queries", len(queries), iteration)
        iteration += 1

    writer.close()
    return bandit_algo, queries


class UCBAlgorithm:
    """Base class for UCB algorithm"""

    def __init__(self, args, data, initial_queries, classification_model, embedding_model, cache):
        """Initialise algorithm."""

        self.args = args
        self.data = data
        self.initial_queries = initial_queries
        self.classification_model = classification_model
        self.embedding_model = embedding_model
        self.cache = cache

        logging.info(f"Loading generation model from {args.ucb_generator_model}..")
        # Currently this script only supports T5
        
        self.generator_model = T5ForConditionalGeneration.from_pretrained(args.ucb_generator_model).to("cuda")
        self.generator_tokenizer = T5TokenizerFast.from_pretrained(args.ucb_generator_tokenizer)
        logging.info("Generator model and tokenizer loaded.")

        logging.info("Loading spacy model..")
        self.nlp = spacy.load(args.ucb_spacy_model)
        logging.info("Spacy model loaded.")

        # initialise nearest neighbor algo for density estimate
        data_texts, data_labels = separate_labels(data)
        data_embeddings = np.array([self.cache["embedding_map"][text] for text in data_texts])
        self.de_nn_algo = NearestNeighbors(algorithm="brute", n_jobs=-1)
        self.de_nn_algo = self.de_nn_algo.fit(data_embeddings)

    def random_query(self, queries):
        """Select random query."""

        return random.choice(list(queries))

    def select_query(self, queries):
        """Select query based on the algorithm"""

        raise NotImplementedError

    def compute_reward(self, generated_queries):
        """Computes rewards based on the generated queries."""

        if len(generated_queries) == 0:
            return 0

        density_estimates, _, epsilons = get_batch_density_estimates(
            self.args,
            generated_queries,
            self.de_nn_algo,
            self.data,
            self.embedding_model,
            self.classification_model,
            cache=self.cache,
        )

        reward = density_estimates.mean().item()  # if density estimate is high, it means more chance of leakage
        reward += 0.1 * mean(epsilons)  # should be higher if sparse -> we want sparse ones
        return reward

    def update(self, query, reward):
        """Updates the parameter of the algorithm"""

        raise NotImplementedError

    def generate_neighbors(self, query):
        """Generated neighbors using the query."""

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
            max_generate=self.args.ucb_neighbors_generate * 2,  # accounting that there will be duplicates
        )

        neighbors = set([x["candidate_leakage"] for x in output])  # make distinct
        if query in neighbors:
            neighbors.remove(query)
        neighbors = list(neighbors)
        if len(neighbors) > self.args.ucb_neighbors_generate:
            neighbors = random.sample(neighbors, self.args.ucb_neighbors_generate)
        return neighbors

    def _embed_queries(self, queries):
        """Internal function to embed the queries."""

        remaining_queries = [query for query in queries if query not in self.cache["embedding_map"]]

        if len(remaining_queries) != 0:
            remaining_embeddings = compute_embeddings(
                self.args, remaining_queries, self.embedding_model, convert_to_numpy=True
            )

            for query, embedding in zip(remaining_queries, remaining_embeddings):
                self.cache["embedding_map"][query] = embedding

        return [self.cache["embedding_map"][query] for query in queries]


class kNNUCBAlgorithm(UCBAlgorithm):
    """Implements i-KNN-UCB algorithm.
    https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0145-0"""

    def __init__(self, args, data, initial_queries, classification_model, embedding_model, cache):
        """Initialise algorithm."""

        super(kNNUCBAlgorithm, self).__init__(args, data, initial_queries, classification_model, embedding_model, cache)

        self.past_rewards = {}

    def select_query(self, queries):
        """Select query based on the algorithm"""

        embeddings = self._embed_queries(queries)
        past_embeddings = self._embed_queries(self.past_rewards.keys())
        nn_algo = NearestNeighbors(algorithm="brute", n_jobs=-1)  # for the algorithm
        nn_algo = nn_algo.fit(past_embeddings)

        # this dictionary maintains scores of all the queries.
        scores = {}

        # check the paper for more information on this part
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
        """Finds k nearest neighbors for the algorithm."""

        n_neighbors = min(len(self.past_rewards), self.args.kucb_k)
        distances, indices = nn_algo.kneighbors([embedding], n_neighbors=n_neighbors)
        distances, indices = distances[0], indices[0]

        past_queries = list(self.past_rewards.keys())
        neighbors = [past_queries[index] for index in indices]
        return neighbors, distances


class LinUCBAlgorithm(UCBAlgorithm):
    """Implements LinUCB algorithm.
    https://arxiv.org/pdf/1003.0146.pdf"""

    def __init__(self, args, data, initial_queries, classification_model, embedding_model, cache):
        """Initialise algorithm."""

        super(LinUCBAlgorithm, self).__init__(args, data, initial_queries, classification_model, embedding_model, cache)
        self.params = {}

    def select_query(self, queries):
        """Select query based on the algorithm"""

        embeddings = self._embed_queries(queries)
        d = embeddings[0].shape[0]

        scores = {}

        for query, embedding in zip(queries, embeddings):
            if query not in self.params:
                A = np.identity(d)
                b = np.zeros((d, 1))
                self.params[query] = (A, b)

            A, b = self.params[query]
            x = embedding.reshape(-1, 1)
            alpha = self.args.lucb_alpha

            theta = np.linalg.inv(A) @ b
            score = np.transpose(theta) @ x
            score += alpha * np.sqrt(np.transpose(x) @ np.linalg.inv(A) @ x)

            scores[query] = score.item()

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

        embedding = self._embed_queries([query])[0]

        A, b = self.params[query]
        x = embedding.reshape(-1, 1)

        A = A + x @ x.transpose()
        b = b + reward * x

        self.params[query] = (A, b)


class ThompsonSampling(UCBAlgorithm):
    """Implements Thompson Sampling with a Beta prior, having means as the density estimates"""

    def __init__(self, args, data, initial_queries, classification_model, embedding_model, cache):
        """Initialize algorithm and calculate priors based on density estimates"""

        super(ThompsonSampling, self).__init__(args, data, initial_queries, classification_model, embedding_model, cache)
        density_estimates, _, _ = get_batch_density_estimates(
            self.args,
            self.initial_queries,
            self.de_nn_algo,
            self.data,
            self.embedding_model,
            self.classification_model,
            cache=self.cache,
        )
        density_estimates = density_estimates.tolist()

        self.reward_distributions = {}
        self.draw_counts = {}
        self.mean_rewards = {}

        for query, p in zip(self.initial_queries, density_estimates):
            rho = pm.Beta.dist('rho', mu=p, sigma=self.args.prior_var)
            self.reward_distributions[query] = rho
            self.mean_rewards[query] = rho.random()
            self.draw_counts[query] = 0.00001

    def select_query(self):
        """Select query based on the algorithm"""

        selected_query = None
        scores = {}
        max_score = -float("inf")
        alpha = self.args.lucb_alpha

        for query in self.initial_queries:
            scores[query] = self.mean_rewards[query] + alpha * np.sqrt(2 * np.log(self.args.time_horizon) / self.draw_counts[query])

        for query, score in scores.items():
            if score > max_score:
                max_score = score
                selected_query = query

        return selected_query

    # Thompson sampling computes reward on the basis of success or failure of the classifier on that example only, while other algorithms compute 
    # on the basis of generating new queries and density estimates. Need to map this properly, such as an increase in density from 
    # previous is a 1 else is 0. However, this is not a Bernoulli likelihood, and the posterior is intractable, so the beta prior maybe be 
    # inappropriate. Else, PyMC3 with NUTS, HMC maybe used to update the prior. Bandit algorithms do not change environment, so are not online.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.getenv("PT_DATA_DIR", "data"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exp_name", type=str, default=str(time.time()))
    parser.add_argument("--labelled_data", type=str, default="hate/labelled_data.txt")
    parser.add_argument("--data_split_ratio", type=int, nargs=3, default=[0.72, 0.08, 0.2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=os.getenv("PT_OUTPUT_DIR", "output"))
    parser.add_argument("--save_cache", action="store_true")
    parser.add_argument("--cache_file", type=str, default="cache.pkl")
    parser.add_argument("--cls_model", type=str, default="roberta-large")
    parser.add_argument("--cls_overwrite_output_dir", action="store_true")
    parser.add_argument("--cls_per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--cls_per_device_eval_batch_size", type=int, default=4096)
    parser.add_argument("--cls_learning_rate", type=float, default=1e-5)
    parser.add_argument("--cls_num_train_epochs", type=int, default=3)
    parser.add_argument("--cls_max_seq_length", type=int, default=40)
    parser.add_argument("--cls_model_save_name", type=str, default="classification_model")
    parser.add_argument("--de_ks", type=int, nargs="+", default=[512])
    parser.add_argument("--de_val_batch_size", type=int, default=2048)
    parser.add_argument("--ucb_save_path", type=str, default="bandit_algo.bin")
    parser.add_argument("--ucb_spacy_model", type=str, default="en_core_web_lg", help="Spacy model to use for T5")
    parser.add_argument("--ucb_neighbors_generate", type=int, default=8, help="Max number of neighbors to generate")
    parser.add_argument("--ucb_queries_generate", type=int, default=100000, help="Total number of queries to generate")
    parser.add_argument("--kucb_alpha", type=float, default=0.001, help="Exploration parameter for knn-ucb")
    parser.add_argument("--kucb_k", type=int, default=128, help="k value for knn-ucb algorithm")
    parser.add_argument("--lucb_alpha", type=float, default=0.01, help="alpha value for linUCB")
    parser.add_argument("--prior_var", type=float, default=0.1, help="Variance of prior Beta distributions in ThompsonSampling")
    parser.add_argument("--time_horizon", type=int, default=10, help="Time horizon or number of rounds played by bandit algorithm")
    parser.add_argument("--ucb_generator_model", type=str, help="T5 model for UCB generator.")
    parser.add_argument("--ucb_generator_tokenizer", type=str, default="t5-large", help="T5 tokenizer for UCB.")
    parser.add_argument("--ucb_queries_fp", type=str, default="generated_queries.txt", help="FP where to save.")
    parser.add_argument("--ucb_algorithm", type=str, default="knn", choices=["knn", "lin"], help="which UCB algorithm.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    random.seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    main(args)
