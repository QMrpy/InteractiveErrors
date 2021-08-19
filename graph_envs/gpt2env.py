import logging
import json
import os
from copy import deepcopy
import gym
import numpy as np
import torch
import networkx as nx
from numba import njit, prange
from torch.utils.data import DataLoader
from stable_baselines3.common import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

from classification import compute_embeddings
from utils import TextDataset


class GPT2RLEnv(gym.Env):

    def __init__(self, args, sentences):
        super().__init__()
        self.seed_sentences = sentences
        self.args = args
        self.output_dict = []
        self.episode_rewards = 0
        self.episode_id = 0
        self.log_dict = []
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.args.top_k,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.args.top_k)
        self.cls_model = AutoModelForSequenceClassification.from_pretrained(self.args.cls_model).to("cuda")
        self.cls_tokenizer = AutoTokenizer.from_pretrained(self.args.cls_model)
        self.gen_model = AutoModelForCausalLM.from_pretrained(self.args.gen_model).to("cuda")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(self.args.gen_model)
        self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token

    def save_outputs(self):
        logging.info(f"Saving sentences and rewards to {self.args.output_file}")
        with open(os.path.join(".", str(self.episode_id) + "_" + self.args.output_file), "w") as file:
            data = {"args": vars(self.args), "output_sentences": self.output_dict}
            json.dump(data, file, indent=2)

        with open(self.args.rewards_log_file, "a") as file:
            file.write(str(self.episode_id) + "\t" + str(self.episode_rewards) + "\n")

        logging.info("Saving RL logs")
        with open(self.args.log_file, "w") as file:
            logs = {"logs": self.log_dict}
            json.dump(logs, file, indent=2)

        logging.info(f"Saving updated networkx graph to {self.args.output_graph_file}")
        json.dump(dict(nodes=[(n, {"sentences": self.graph.nodes[n]["sentences"], "cls_embeddings": self.graph.nodes[n]["cls_embeddings"], "plot_labels": self.graph.nodes[n]["plot_labels"]}) for n in self.graph.nodes()],
                edges=[e for e in list(self.graph.edges)]),
                open(self.args.output_graph_file, "w"), indent=2)


    def step(self, action):
        self.current_step += 1

        selected_sentence = self.topk_inputs[action]         
        logging.info(f"Action selected is sentence {action}")                     
        logging.info(f"Selected sentence : {selected_sentence}")

        centrality = nx.eigenvector_centrality(self.graph)
        node_centrality = centrality[self.action_to_indexes[action]]

        input_prefix = ""
        if len(selected_sentence) > self.args.prefix_length:
            input_prefix = " ".join(selected_sentence.split()[:self.args.prefix_length])
        else:
            input_prefix = selected_sentence
        logging.info(f"Input prefix is {input_prefix}")

        input_ids = self.gen_tokenizer(input_prefix, return_tensors="pt", padding=True, truncation=True, max_length=128, add_special_tokens=True).input_ids.to("cuda")
        outputs = self.gen_model.generate(input_ids=input_ids, min_length=5, max_length=20, do_sample=True, top_k=50, top_p=0.98, num_return_sequences=15)
        gen_sentences = [self.gen_tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in outputs]
        
        gen_sentences = set(gen_sentences)
        for i in self.inputs:
            if i in gen_sentences:
                gen_sentences.remove(i)
        gen_sentences = list(gen_sentences)

        logging.info(f"{len(gen_sentences)} sentences generated")
        if len(gen_sentences) == 0:
            logging.info("Reward received is 0")
            logging.info(f"current_step = {self.current_step}")

            done = 0
            if self.current_step >= self.args.horizon:
                self.save_outputs()
                done = 1

            return self.logits, 0, done, {}

        for i in range(len(gen_sentences)):
            logging.info(f"Generated sentence {i} : {gen_sentences[i]}")
        
        gen_embeddings = compute_embeddings(self.args, gen_sentences, {"model": self.cls_model, "tokenizer": self.cls_tokenizer}, True).tolist()
        self.similarities = update_similarities(gen_sentences, gen_embeddings, self.inputs, self.embeddings, self.similarities, self.args)
        self.graph = update_networkx_graph(self.graph, gen_sentences, gen_embeddings, self.inputs, self.similarities, self.args)

        for i in range(len(gen_sentences)):
            self.inputs.append(gen_sentences[i])            
            self.embeddings.append(gen_embeddings[i])
        
        text_dataset = TextDataset(self.args, self.cls_tokenizer, self.inputs)
        input_dataloader = DataLoader(text_dataset, batch_size=self.args.batch_size)

        obs = []
        for batch in input_dataloader:
            batch = batch["input_ids"].to("cuda")
            with torch.no_grad():
                batch_obs = self.cls_model(batch).logits                             
            batch_obs = batch_obs[:, 0].detach().cpu().data.numpy()
            obs.extend(batch_obs.tolist())
        obs = torch.Tensor(obs)
     
        centrality = nx.eigenvector_centrality(self.graph)

        node_rewards = 0
        reward = np.zeros(shape=len(gen_sentences))

        for i in range(len(gen_sentences)):
            if self.logits[action] < obs[i + len(self.inputs) - len(gen_sentences)]:
                reward[i] += 1
                node_rewards += 1
            if node_centrality < centrality[i + len(self.inputs) - len(gen_sentences)]:
                reward[i] += 1
                node_rewards += 1

        self.episode_rewards += node_rewards
        logging.info(f"Reward received is {node_rewards}")

        if node_rewards > 0:
            for i in range(len(gen_sentences)):
                self.output_dict.append(
                    {
                        "selected_sentence": selected_sentence,
                        "input_prefix": input_prefix,
                        "initial_logit": str(self.logits[action]),
                        "node_rewards": str(node_rewards),
                        "node_centrality": str(node_centrality),
                        "generated_sentence": gen_sentences[i],
                        "gen_sentence_logit": str(obs[i + len(self.inputs) - len(gen_sentences)].numpy()),
                        "gen_sentence_reward": str(reward[i]),
                        "gen_sentence_centrality": str(centrality[i + len(self.inputs) - len(gen_sentences)]),
                    }
                )

        logging.info(f"current_step = {self.current_step}")
        
        obs, indices = torch.sort(obs, descending=True)
        obs, indices = obs[:self.args.top_k], indices[:self.args.top_k]
        
        obs = obs.numpy()
        self.logits = obs
        self.topk_inputs = [self.inputs[i] for i in indices]
        self.action_to_indexes = indices.numpy()

        logs = logger.get_log_dict()
        self.log_dict.append(
            {
                "n_updates": logs["train/n_updates"], 
                "value_loss": logs["train/value_loss"], 
                "policy_loss": logs["train/policy_loss"],
                "entropy_loss": logs["train/entropy_loss"],
                "explained_variance": logs["train/explained_variance"],
                "learning_rate": logs["train/learning_rate"],
            }
        )
        
        done = 0
        if self.current_step >= self.args.horizon:
            self.save_outputs()
            done = 1
        
        return obs, node_rewards, done, {}

    def reset(self):
        self.inputs = deepcopy(self.seed_sentences)
        self.embeddings = compute_embeddings(self.args, self.inputs, {"model": self.cls_model, "tokenizer": self.cls_tokenizer}, True, True).tolist()
        self.similarities = compute_similarities(self.inputs, self.embeddings, self.args)
        self.graph = construct_networkx_graph(self.similarities, self.inputs, self.embeddings, self.args)
        
        logging.info(f"Saving networkx graph to {self.args.input_graph_file}")
        json.dump(dict(nodes=[(n, {"sentences": self.graph.nodes[n]["sentences"], "cls_embeddings": self.graph.nodes[n]["cls_embeddings"], "plot_labels": self.graph.nodes[n]["plot_labels"]}) for n in self.graph.nodes()],
                    edges=[e for e in list(self.graph.edges)]),
                    open(self.args.input_graph_file, "w"), indent=2)
        
        self.current_step = 0
        self.output_dict = []
        self.episode_rewards = 0
        self.episode_id += 1

        text_dataset = TextDataset(self.args, self.cls_tokenizer, self.inputs)
        input_dataloader = DataLoader(text_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        obs = []
        for batch in input_dataloader:
            batch = batch["input_ids"].to("cuda")
            with torch.no_grad():
                batch_obs = self.cls_model(batch).logits                             
            batch_obs = batch_obs[:, 0].detach().cpu().data.numpy()
            obs.extend(batch_obs.tolist())
        obs = torch.Tensor(obs)

        obs, indices = torch.sort(obs, descending=True)
        obs, indices = obs[:self.args.top_k], indices[:self.args.top_k]

        obs = obs.numpy()
        self.logits = obs
        self.topk_inputs = [self.inputs[i] for i in indices]
        self.action_to_indexes = indices.numpy()

        return obs


def compute_similarities(sentences, embeddings, args):        
    normalized_embeddings = []
    eps = 0.000001
    
    logging.info("Calculating node attributes")
    normalized_embeddings = [embeddings[i] / (eps + np.linalg.norm(embeddings[i])) for i in range(len(sentences))]        
    normalized_embeddings = np.array(normalized_embeddings)

    @njit(parallel=True)
    def optimized(embeddings):
        """ Computes the similarity calculations in an optimized manner using numba njit """
        similarities = np.zeros(shape=(len(embeddings), len(embeddings)))
        
        for i in prange(len(embeddings)):
            for j in prange(len(embeddings)):
                similarities[i][j] = np.dot(embeddings[i], np.transpose(embeddings[j]))

        return similarities

    if args.similarity_algo == "cosine":
        logging.info("Calculating edge attributes or sentence similarity scores")
        similarities = optimized(normalized_embeddings)
    else:
        raise ValueError("Invalid similarity algorithm")

    return similarities


def construct_networkx_graph(similarities, sentences, embeddings, args):
    edges = []
   
    if args.similarity_algo == "cosine":
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if similarities[i][j] >= args.sim_threshold:
                    edges.append((i, j))                                
        
    else:
        raise ValueError("Invalid similarity algorithm")
                    
    graph = nx.Graph()
    embeddings = np.squeeze(embeddings)
    graph.add_nodes_from([(v, {"sentences": sentences[v], "cls_embeddings": embeddings[v].tolist(), "plot_labels": 0}) for v in range(len(sentences))])
    graph.add_edges_from(edges)
    logging.info("Constructed graph")
    
    return graph


def update_similarities(gen_sentences, gen_embeddings, sentences, embeddings, similarities, args):
    logging.info(f"Size of similarity matrix before updation is {len(similarities)} x {len(similarities)}")
   
    if args.similarity_algo == "cosine":
        eps = 0.000001
        normalized_embeddings = []
        normalized_gen_embeddings = []

        x = np.zeros(shape=(len(gen_sentences), len(sentences)))
        similarities = np.vstack((similarities, x))
        y = np.zeros(shape=(len(sentences) + len(gen_sentences), len(gen_sentences)))
        similarities = np.hstack((similarities, y))
        
        for i in embeddings:
            normalized_embeddings.append(i / (eps + np.linalg.norm(i)))

        for i in gen_embeddings:
            normalized_gen_embeddings.append(i / (eps + np.linalg.norm(i)))

        for i in range(len(gen_sentences)):
            for j in range(len(sentences)):
                similarities[i + len(sentences)][j] = np.dot(normalized_gen_embeddings[i], np.transpose(normalized_embeddings[j]))
                similarities[j][i + len(sentences)] = np.dot(normalized_embeddings[j], np.transpose(normalized_gen_embeddings[i]))

        for i in range(len(gen_sentences)):
            for j in range(len(gen_sentences)):
                similarities[i + len(sentences)][j + len(sentences)] = np.dot(normalized_gen_embeddings[i], np.transpose(normalized_gen_embeddings[j]))

        logging.info(f"Size of similarity matrix after updation is {len(similarities)} x {len(similarities)}")
        
    else:
        raise ValueError("Invalid similarity algorithm")

    return similarities


def update_networkx_graph(graph, gen_sentences, gen_embeddings, sentences, similarities, args):
    new_edges = []

    if args.similarity_algo == "cosine":
        for i in range(len(gen_sentences)):
            for j in range(len(sentences)):
                if similarities[i + len(sentences)][j] >= args.sim_threshold:
                    new_edges.append((i + len(sentences), j))
                if similarities[j][i + len(sentences)] >= args.sim_threshold:
                    new_edges.append((j, i + len(sentences)))

        for i in range(len(gen_sentences)):
            for j in range(len(gen_sentences)):
                if similarities[i + len(sentences)][j + len(sentences)] >= args.sim_threshold:
                    new_edges.append((i + len(sentences), j + len(sentences)))
                if similarities[j + len(sentences)][i + len(sentences)] >= args.sim_threshold and i != j:
                    new_edges.append((j + len(sentences), i + len(sentences)))

    else:
        raise ValueError("Invalid similarity algorithm")

    gen_embeddings = np.squeeze(gen_embeddings)
    graph.add_nodes_from([(v + len(sentences), {"sentences": gen_sentences[v], "cls_embeddings": gen_embeddings[v].tolist(), "plot_labels": 1}) for v in range(len(gen_sentences))])
    graph.add_edges_from(new_edges)
    logging.info("Updated graph")
    
    return graph



"""
Comments:
1. Use GPT2 Large
2. Do something with the number of neighbors, etc., of the selected sentence before and after updating graph
as reward  ---- done
3. Handle observations properly. top-k observations can cause problems of losing information  ---- done
4. Maybe keep a map so as to know which sentences were generated corresponding to which action  ---- done
5. Get stability of graph
6. Visualize graph and movement of sentences and embeddings  ---- partially done
7. Look into variable action and observation spaces
8. Record the sentences before and after a step  ---- done
9. Make a latent model for generation from neighborhood, z = CLS(BERT(x)), x' = GPT2(z'), z' = z + e
10. Save more information to graph such as plot labels, leakages, similarities, rewards, etc.  ---- done
11. Remove the similarities dictionary, and use a numpy array. The dictionary is not at all required.
12. edges could have been a shared buffer, in which case, we could have appended parallely, or will the different
workers see different lengths of the same shared buffer? Parallel append is a hard problem
13. networkx is the main bottleneck of the code. Need a parallel graph library
"""