import logging
import json
import os
import re
import random
from copy import deepcopy
import gym
import numpy as np
import torch
import spacy
import itertools
import networkx as nx
from nltk.translate.bleu_score import sentence_bleu 
from numba import njit, prange
from torch.utils.data import DataLoader
from stable_baselines3.common import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

from classification import compute_embeddings
from utils import TextDataset
from bart import convert_to_bart_mlm_format
from graph_envs.bartenv import BARTRLEnv, compute_similarities, construct_networkx_graph, update_similarities, update_networkx_graph



class SynthBARTRLEnv(BARTRLEnv):

    def __init__(self, args, sentences):
        super().__init__(args, sentences)
        self.cls_model = AutoModelForSequenceClassification.from_pretrained(self.args.cls_model).to("cuda")
        self.cls_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.seed_ancestors = {sen: i for i,sen in enumerate(self.seed_sentences)}
        # TODO: 1-4 bleu overlap with ancestor seed sentence.
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.args.top_k,4), dtype=np.float32)
        self.sim_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
        self.nlp = spacy.load(args.spacy_model)
        self.use_sim_model = args.use_sim_model
        

    def step(self, action):
        self.current_step += 1

        selected_sentence = self.topk_inputs[action]         
        logging.info(f"Action selected is sentence {action}")                     
        logging.info(f"Selected sentence : {selected_sentence}")

        centrality = nx.eigenvector_centrality(self.graph)
        node_centrality = centrality[self.action_to_indexes[action]]

        gen_sentences, masked_sentences = mask_generate(self.nlp, selected_sentence, self.gen_model, self.sim_model, self.gen_tokenizer)
        gen_sentences = set(gen_sentences)
        gen_sentences = gen_sentences.difference(set(self.inputs))
        gen_sentences = list(gen_sentences)
        
        # Update seed ancestors: sent -> orig. seed leakage id 
        root_parent_id = self.seed_ancestors[selected_sentence]
        for i,sent in enumerate(gen_sentences):           
            self.seed_ancestors[sent] = root_parent_id

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

        ############################ Observations
        # Obs: 4-tuple [1-gram, 2-gram, 3-gram, 4-gram] similarity to ancestor seed sentence.
        obs = []
        for j in range(len(self.inputs)):
            parent_id = self.seed_ancestors[self.inputs[j]]
            if parent_id == j:
                _obs =  [0., 0., 0., 0.]
            else:
                _obs = get_bleu_vector(self.inputs[parent_id], self.inputs[j])
            obs.append(_obs)
        obs = torch.Tensor(obs)     
        centrality = nx.eigenvector_centrality(self.graph)

        node_rewards = 0
        reward = np.zeros(shape=len(gen_sentences))        
        # 1. Topic distribution should align. -> LDA could suffer from out-of-vocab issue. So, perplexity is better.
        # 2. (TODO) perplexity wrt a query model should be low. 
        # 3. proxy reward - bleu scores. Paraphrase LM seems better
        # 4. centrality seems to prefer bigger sentences.
        print(f"Parent id={root_parent_id}, action={action}")
        print(f"Len gen sent={len(gen_sentences)}, Current inputs={len(self.inputs)}")
        
        root_parent_text = self.inputs[root_parent_id]
        action_bleu_2g = 0
        is_action_seed =  True
        if root_parent_text != selected_sentence: # execute if its not a seed sentence
            is_action_seed =  False
            action_weighted_bleu = weighted_sum_bleu(root_parent_text, selected_sentence)
            for j in range(len(self.topk_inputs)):
                if j != action:
                    j_ref_text = self.inputs[self.seed_ancestors[self.topk_inputs[j]]]
                    if j_ref_text != self.topk_inputs[j] and weighted_sum_bleu(j_ref_text, self.topk_inputs[j]) < action_weighted_bleu:
                        node_rewards += 1 
            node_rewards = node_rewards/len(self.topk_inputs)

        gen_reward = 0
        if self.use_sim_model and not is_action_seed:
            action_sentence_embedding = self.sim_model.encode([selected_sentence], convert_to_tensor=True)
            parent_sentence_embedding = self.sim_model.encode([root_parent_text], convert_to_tensor=True)
            gen_sentence_embeddings = self.sim_model.encode(gen_sentences, convert_to_tensor=True)
            action_sim_cosine_score = util.pytorch_cos_sim(parent_sentence_embedding, action_sentence_embedding).cpu().numpy()
            gen_similarity_cosine_scores = util.pytorch_cos_sim(parent_sentence_embedding, gen_sentence_embeddings).cpu().numpy()
            for i in range(0,len(gen_sentences)):
                if gen_similarity_cosine_scores[0,i] > action_sim_cosine_score[0,0]:
                    reward[i] += gen_similarity_cosine_scores[0,i]
                    gen_reward += 1
        
        action_weighted_bleu = weighted_sum_bleu(root_parent_text, selected_sentence)
        for i in range(len(gen_sentences)):
            if node_centrality < centrality[i + len(self.inputs) - len(gen_sentences)]:
                reward[i] += 1
                gen_reward += 1
            # TODO: This is a heuristic for an action if its a seed sentence. REMOVE LATER
            if is_action_seed:
                _2_gram = sentence_bleu([selected_sentence.split()], gen_sentences[i].split(), weights=(0, 1, 0, 0))
                _3_gram = sentence_bleu([selected_sentence.split()], gen_sentences[i].split(), weights=(0, 0, 0.5, 0.5))
                if _2_gram > _3_gram:
                    reward[i] += (_2_gram + _3_gram)/2.0
                    gen_reward += 1
            if not self.use_sim_model and not is_action_seed:               
                gen_weighted_bleu = weighted_sum_bleu(root_parent_text, gen_sentences[i])
                if action_weighted_bleu < gen_weighted_bleu:
                    reward[i] += gen_weighted_bleu
                    gen_reward += 1
            
        node_rewards = node_rewards + gen_reward/len(gen_sentences)
        ############################ Change here
        
        self.episode_rewards += node_rewards
        logging.info(f"Reward received is {node_rewards}")

        if node_rewards > 0:
            for i in range(len(gen_sentences)):
                self.output_dict.append(
                    {
                        "selected_sentence": selected_sentence,
                        "masked_sentence": masked_sentences[i],
                        "node_rewards": str(node_rewards),
                        "node_centrality": str(node_centrality),
                        "generated_sentence": gen_sentences[i],
                        "gen_sentence_logit": str(obs[i + len(self.inputs) - len(gen_sentences)].numpy()),
                        "gen_sentence_reward": str(reward[i]),
                        "gen_sentence_centrality": str(centrality[i + len(self.inputs) - len(gen_sentences)]),
                    }
                )

        logging.info(f"current_step = {self.current_step}")
        
        ############################ Change here
        _, indices = torch.sort(obs, dim=0, descending=True) # sort on 2-gram
        indices = indices[:self.args.top_k,0]
        obs = torch.index_select(obs, 0, indices)
        # obs, indices = torch.sort(obs, dim=1, descending=True)
        # obs, indices = obs[:self.args.top_k], indices[:self.args.top_k]
        
        obs = obs.numpy()
        self.logits = obs
        self.topk_inputs = [self.inputs[i] for i in indices]
        ############################ Change here
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

        ############################ Change here
        obs = []
        for j in range(len(self.inputs)):
            parent_id = self.seed_ancestors[self.inputs[j]]
            if parent_id == j:
                _obs = [0., 0., 0., 0.]
            else:
                _obs = get_bleu_vector(self.inputs[parent_id],self.inputs[j])
            obs.append(_obs)
        obs = torch.Tensor(obs)
        ############################ Change here

        # import pdb
        # pdb.set_trace()
        _, indices = torch.sort(obs, dim=0, descending=True) # sort on 2-gram
        indices = indices[:self.args.top_k,0]
        obs = torch.index_select(obs, 0, indices)
        # obs, indices = obs[indices[:,0]][:self.args.top_k], indices[:self.args.top_k,0]

        obs = obs.numpy()
        self.logits = obs
        self.topk_inputs = [self.inputs[i] for i in indices]
        self.action_to_indexes = indices.numpy()

        return obs


def remove_adjacent_masks(text):
    regex = r'\b(\w+)(?:\W+\1\b)+' 
    return re.sub(regex, r'\1', text, flags=re.IGNORECASE)


def mask_pos_tags(nlp_spacy, sentence):
    pos_tags = ["VERB", "ADJ", "ADV", "NOUN", "PROPN"]

    tokens = nlp_spacy(sentence)
    masks = ["".join(seq) for seq in itertools.product("01", repeat=min(12, len(tokens)))]
    masked_sentences = set()

    for mask in masks:
        masked_sentence = []

        for i in range(len(tokens)):
            if i < 12 and tokens[i].pos_ in pos_tags and mask[i] == "1":
                masked_sentence.append("<mask>")
            else:
                masked_sentence.append(tokens[i].text)
        
        masked_sentence = ' '.join(masked_sentence)
        masked_sentence = remove_adjacent_masks(masked_sentence)
        if masked_sentence.count("<mask>") > 2 or masked_sentence.count("<mask>") == 0:
            continue
        masked_sentences.add(masked_sentence)

    masked_sentences = list(masked_sentences)

    return masked_sentences


def mask_generate(nlp_spacy, sentence, gen_model, sim_model, tokenizer):
    masked_templates = []

    logging.info(f"Generating mask templates for sentence: {sentence}")
    masked_sentences = mask_pos_tags(nlp_spacy, sentence)
    sentence_embedding = sim_model.encode([sentence], convert_to_tensor=True)
    
    # HACK: Randomize and select a few masks. Not all.
    random.shuffle(masked_sentences)
    masked_sentences = masked_sentences[:min(25, len(masked_sentences))]
        
    for masked_sentence in masked_sentences:
        logging.info(f"Generating from masked template sentence: {masked_sentence}")
        input_ids = tokenizer(masked_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids.to("cuda")

        outputs = gen_model.generate(input_ids, num_return_sequences=10, top_k=50, top_p=0.99, do_sample=True)
        output_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        output_sentences = list(set(output_sentences))

        gen_sentence_embeddings = sim_model.encode(output_sentences, convert_to_tensor=True)

        similarity_cosine_scores = util.pytorch_cos_sim(sentence_embedding, gen_sentence_embeddings)
        
        mask_score = torch.mean(similarity_cosine_scores).item()
        logging.info(f"Score of masked template based on sentence similarities is {mask_score}")

        mask_std = torch.std(similarity_cosine_scores).item()

        masked_templates.append({"masked_sentence": masked_sentence, "mask_score": mask_score, "mask_standard_deviation": mask_std, "generated_sentences": output_sentences})

    masked_templates = sorted(masked_templates, key=lambda x: x["mask_score"], reverse=True)
    sentences = []
    masked_sentences = []
    for tmplt in masked_templates:
        if tmplt["mask_score"] > 0.99 and tmplt["mask_score"] < 0.6:
            continue
        sentences.extend(tmplt["generated_sentences"])
        masked_sentences.extend([tmplt['masked_sentence'] for _ in range(len(tmplt["generated_sentences"]))])
        if len(sentences) > 10:
            break
    assert len(sentences) == len(masked_sentences)
    logging.info("-----------------------------------------------------------------------")
    return sentences, masked_sentences

def get_bleu_vector(reference_text, hypothesis_text):
    _1_gram = sentence_bleu([reference_text.split()], hypothesis_text.split(), weights=(1.0, 0, 0, 0))
    _2_gram = sentence_bleu([reference_text.split()], hypothesis_text.split(), weights=(0.5, 0.5, 0, 0))
    _3_gram = sentence_bleu([reference_text.split()], hypothesis_text.split(), weights=(0.33, 0.33, 0.33, 0))
    _4_gram = sentence_bleu([reference_text.split()], hypothesis_text.split())    
    return [_1_gram, _2_gram, _3_gram, _4_gram]

def weighted_sum_bleu(reference_text, hypothesis_text):
    return (3 * sentence_bleu([reference_text.split()], hypothesis_text.split(), weights=(0.5, 0.5, 0, 0)) + 2 * sentence_bleu([reference_text.split()], hypothesis_text.split(), weights=(0.33, 0.33, 0.33, 0)) + sentence_bleu([reference_text.split()], hypothesis_text.split()))/6.0