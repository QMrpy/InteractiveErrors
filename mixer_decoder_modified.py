"""WARNING: Incomplete Code"""

import argparse
import logging
import random
import os
import warnings
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, RobertaForSequenceClassification, RobertaTokenizer

from sampling import pointwise_gaussian_sampling, gaussian_process_sampling


class Mixer(nn.Module):           

    def __init__(self, cls_hidden_dim, gen_hidden_dim, cls_num_layers, gen_num_layers):
        super().__init__()
        self.layer1 = nn.Linear(2 * cls_hidden_dim * cls_num_layers, cls_hidden_dim * cls_num_layers, bias=True)
        self.activation1 = nn.GELU()
        self.layer2 = nn.Linear(cls_hidden_dim * cls_num_layers, gen_hidden_dim * gen_num_layers, bias=True)
        self.activation2 = nn.GELU()

    def forward(self, original_embedding, sampled_embedding):
        flattened_original = torch.flatten(original_embedding)
        flattened_sampled = torch.flatten(sampled_embedding)
        x = torch.cat([flattened_original, flattened_sampled]).view(-1)
        print(x.shape)
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = torch.reshape(x, original_embedding.shape)

        return x
 

def train_model(args, data, cls_model, cls_tokenizer, gen_model, gen_tokenizer, epochs=5, learning_rate=5e-5):
    logging.info("Sampling nearby embeddings (hidden_states)")
    original_hidden_states, sampled_hidden_states = [], []

    if args.sampling_algo == "gaussian_process":
        original_hidden_states, sampled_hidden_states = gaussian_process_sampling(args, data, cls_model, cls_tokenizer)

    elif args.sampling_algo == "pointwise_gaussian":
        original_hidden_states, sampled_hidden_states = pointwise_gaussian_sampling(args, data, cls_model, cls_tokenizer)

    else:
        raise NotImplementedError("This sampling algorithm has not been implemented")
    
    original_hidden_states, sampled_hidden_states = torch.tensor(original_hidden_states).float().to("cuda"), torch.tensor(sampled_hidden_states).float().to("cuda")
    
    mixer = Mixer(cls_model.config.hidden_size, 
    gen_model.config.hidden_size, 
    cls_model.config.num_hidden_layers, 
    gen_model.config.n_layer
    ).to("cuda")

    params = list(mixer.parameters()) + list(gen_model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    logging.info("Training mixer decoder model")
    for _ in range(epochs):
        epoch_loss = 0.0

        for i in tqdm(range(len(data))):
            optimizer.zero_grad()

            past_keys = mixer(original_hidden_states[i], sampled_hidden_states[i])
            past_values = past_keys

            raise NotImplementedError("Code has not been implemented yet. \
            Tuple and Tensor shapes and devices are causing problems.")

        logging.info(f"epoch_loss: {epoch_loss / len(data)}")


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.simplefilter(action='ignore', category=FutureWarning)
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    with open(args.queries_file, "r") as f:
        data = f.read().splitlines()
   
    logging.warning("AutoModelForSequenceClassification doesn't have use_cache or \
    doesn't return past_key_values. Using output_hidden_states")

    cls_tokenizer = RobertaTokenizer.from_pretrained(args.cls_model)
    cls_model = RobertaForSequenceClassification.from_pretrained(args.cls_model).to("cuda")

    gen_tokenizer = GPT2Tokenizer.from_pretrained(args.gen_model)
    gen_model = GPT2LMHeadModel.from_pretrained(args.gen_model).to("cuda")
    
    trained_model = train_model(args, data, cls_model, cls_tokenizer, gen_model, gen_tokenizer)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_file", type=str, help="Path of queries dataset")
    parser.add_argument("--cls_model", type=str, default="roberta-large", help="Path or name of classifier model")
    parser.add_argument("--gen_model", type=str, default="gpt2-medium", help="Path or name of generator model")
    parser.add_argument("--noise_var", type=float, default=0.001, help="Noise variance of multivariate normal distribution")
    parser.add_argument("--sampling_algo", type=str, default="pointwise_gaussian", help="Sampling algorithm to use")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("filelock").setLevel(logging.ERROR)

    main(args)