""" WARNING: Untested Code"""

import argparse
import logging
import os
import random
import torch
import numpy as np
import gym
import ray
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print

from graph_envs.gpt2env import GPT2RLEnv


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    with open(args.input_file, 'r') as f:
        leakages = f.read().splitlines()

    sampled_indices = np.random.choice(np.arange(len(leakages)), size=min(args.samples, len(leakages)), replace=False)
    sampled_data = [leakages[i] for i in sampled_indices]  

    env = GPT2RLEnv(sentences=sampled_data, args=args)
    
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["num_workers"] = 4
    config["num_envs_per_worker"] = 2
    config["seed"] = RANDOM_SEED
    config["framework"] = "torch"
    config["remote_worker_envs"] = True

    trainer = ppo.PPOTrainer(env=env, config=config)

    for _ in range(args.steps):
        result = trainer.train()
        logging.info(pretty_print(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Full path of dataset")
    parser.add_argument("--output_file", type=str, help="Full path of output sentences and rewards")
    parser.add_argument("--log_file", type=str, default="./rl_log_file.json", help="Path of file to save logs of RL algorithm")
    parser.add_argument("--rewards_log_file", type=str, default="./episode_rewards_log_file", help="Path of file where to log episodic rewards")
    parser.add_argument("--input_graph_file", type=str, default="./rl-input-graph.json", help="Full path of where to save the initial graph")
    parser.add_argument("--output_graph_file", type=str, default="./rl-output-graph.json", help="Full path of where to save the final graph")
    parser.add_argument("--samples", type=int, default=200, help="Size of dataset to sample")
    parser.add_argument("--cls_model", type=str, default="roberta-base", help="Path or name of classifier model")
    parser.add_argument("--gen_model", type=str, default="gpt2-medium", help="Path or name of generator model")
    parser.add_argument("--prefix_length", type=int, default=2, help="Prompt length of autoregressive generator model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to send data in device")
    parser.add_argument("--top_k", type=int, default=50, help="Only consider these many top observations and actions. This is also equal to size of observation and action spaces")
    parser.add_argument("--steps", type=int, default=400, help="Required by OpenAI Gym. Total number of samples to train on")
    parser.add_argument("--horizon", type=int, default=200, help="Total timesteps for each episode of RL algorithm")
    parser.add_argument("--similarity_algo", type=str, default="cosine", help="Similarity algorithm to find sentence simiarities")
    parser.add_argument("--sim_threshold", type=float, default=0.6, help="Threshold of similarity for two nodes to be connected by an edge")
    parser.add_argument("--cls_per_device_eval_batch_size", type=int, default=128, help="Keep same as batch size")
    parser.add_argument("--cls_max_seq_length", type=int, default=128, help="Truncate all sequences to this length")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args)

