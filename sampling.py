"""
WARNING: Gaussian Processes function is untested, though it is expected to 
perform as desired
"""

import torch
import numpy as np
import pymc3 as pm
from tqdm import tqdm


def compute_hidden_states(data, model, tokenizer, all_layers=False):
    all_hidden_states = []

    if all_layers:
        for sentence in tqdm(data):
            hidden_states = np.zeros((model.config.num_hidden_layers, model.config.hidden_size))

            input_ids = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids.to("cuda")
            hidden_states_tuple = model(input_ids, output_hidden_states=True).hidden_states[1:]
            
            for layer in range(len(hidden_states_tuple)):
                hidden_states[layer] = hidden_states_tuple[layer].detach().cpu().numpy()[:, 0, :].squeeze()

            all_hidden_states.append(hidden_states)
    else:
        for sentence in tqdm(data):
            input_ids = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids.to("cuda")
            hidden_states = model(input_ids, output_hidden_states=True).hidden_states[-1].detach().cpu().numpy()[:, 0, :].squeeze()

            all_hidden_states.append(hidden_states)

    return np.array(all_hidden_states)
    

def gaussian_process_sampling(args, data, model, tokenizer):
    noise_var = args.noise_var
    N = len(data)

    hidden_states = compute_hidden_states(data, model, tokenizer, all_layers=True)
    hidden_states_t = np.einsum('ijk->jki', hidden_states)

    mean = pm.gp.mean.Zero()
    cov = pm.gp.cov.ExpQuad(1, ls=2)

    sampled_hidden_states_t = []
    for layer in tqdm(hidden_states_t):
        dim_hidden_states_t = []

        for dim in tqdm(layer):
            X = np.expand_dims(dim, axis=1)

            gp_prior = np.random.multivariate_normal(mean(X).eval(), cov(X).eval() + 1e-6 * np.eye(N), 1).squeeze()
            obs = gp_prior + noise_var * np.random.randn(N)
        
            dim_hidden_states_t.append(obs.tolist())
        
        sampled_hidden_states_t.append(dim_hidden_states_t)
    
    sampled_hidden_states_t = np.array(sampled_hidden_states_t)
    sampled_hidden_states = np.einsum('ijk->kij', sampled_hidden_states_t)

    assert sampled_hidden_states.shape == (len(data), model.config.num_hidden_layers, model.config.hidden_size), "Sampled hidden_states are not of the proper shape"

    return hidden_states, sampled_hidden_states


def pointwise_gaussian_sampling(args, data, model, tokenizer):
    noise_var = args.noise_var
    N = len(data)

    hidden_states = compute_hidden_states(data, model, tokenizer, all_layers=True)
    hidden_states_t = np.einsum('ijk->jik', hidden_states)
    
    hidden_size = model.config.hidden_size
    cov = np.zeros((hidden_size, hidden_size))
    np.fill_diagonal(cov, noise_var)

    sampled_hidden_states = []
    for layer in tqdm(hidden_states_t):
        layer_sampled_hidden_states = []

        for sentence_hidden_states in layer:
            layer_sampled_hidden_states.append(np.random.multivariate_normal(sentence_hidden_states, cov))

        sampled_hidden_states.append(layer_sampled_hidden_states)

    sampled_hidden_states = np.array(sampled_hidden_states)
    sampled_hidden_states = np.einsum('ijk->jik', sampled_hidden_states)

    assert sampled_hidden_states.shape == (len(data), model.config.num_hidden_layers, model.config.hidden_size), "Sampled hidden_states are not of the proper shape"

    return hidden_states, sampled_hidden_states

