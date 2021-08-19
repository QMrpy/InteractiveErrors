import argparse
import os
import random
import logging
import copy
import json
from typing import List
import numpy as np
from numpy.lib.function_base import gradient
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification


def restore_list(input_list: List, indices_list: List):
    sequence = [0 for _ in range(len(indices_list))]
    for x in range(len(indices_list)):
        sequence[indices_list[x]] = input_list[x]

    return sequence


def rank_sentences(sentences: List[str], args):
    """ Used to determine which sentences to perturb. Insert custom algorithm here. """

    return sentences


def rank_words(sentence, model, tokenizer, args):                                                 
    """ Used to determine which words to perturb. Insert custom algorithm here. """

    if args.word_rank_algo == "word_attentions":

        input_ids = tokenizer(sentence, truncation=True, max_length=128, return_tensors='pt').input_ids.to('cuda')
        
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)                                                                                            
            attentions = torch.sum(outputs.attentions[-1].squeeze(), dim=0).diag().tolist()                          
        input_ids = input_ids.squeeze().tolist()      
        
        rank_ids = sorted([(attentions[i], input_ids[i]) for i in range(len(input_ids))], reverse=True)                     
        indices = sorted(range(len(input_ids)), reverse=True, key=lambda x: attentions[x])
        rank_ids = [r[1] for r in rank_ids]
        
        return rank_ids, indices
    
    elif args.word_rank_algo == "word_gradients":

        input_ids = tokenizer(sentence, truncation=True, max_length=128, return_tensors='pt').input_ids.to('cuda')
        logit = model(input_ids).logits.squeeze()[0]
        gradients = torch.autograd.grad(outputs=logit, inputs=model.parameters(), retain_graph=True, create_graph=True)
        
        input_ids = input_ids.squeeze().tolist() 
        word_gradients = [torch.norm(gradients[0][i]) for i in input_ids]
       
        rank_ids = sorted([(word_gradients[i], input_ids[i]) for i in range(len(input_ids))], reverse=True)                     
        indices = sorted(range(len(input_ids)), reverse=True, key=lambda x: word_gradients[x])
        rank_ids = [r[1] for r in rank_ids]
        
        return rank_ids, indices

    else:
        raise ValueError("Invalid word ranking algorithm")


def leakage_criterion(sentence, new_sentence, model, tokenizer, args):    

    accept = False
    if args.acceptance_algo == "classifier_score":
        
        sentence_ids = tokenizer.encode(sentence, return_tensors='pt').to('cuda')           
        new_sentence_ids = tokenizer.encode(new_sentence, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            classification_score = model(sentence_ids).logits.squeeze()[0]
            new_classification_score = model(new_sentence_ids).logits.squeeze()[0]
        
        if new_classification_score > classification_score:
            accept = True

    else:
        raise ValueError("Invalid acceptance algorithm")

    return accept


def generate_candidate_leakages(sentences, cls_model, mask_model, cls_tokenizer, mask_tokenizer, args):
    candidate_leakage_dicts = []

    sentences = rank_sentences(sentences, args)

    for sentence in sentences:
        logging.info(f"Current sentence is {sentence}")
        ranked_word_ids, indices = rank_words(sentence, cls_model, cls_tokenizer, args)

        for i in range(len(ranked_word_ids)):    
            if ranked_word_ids[i] == mask_tokenizer.encode(mask_tokenizer.cls_token)[1] or ranked_word_ids[i] == mask_tokenizer.encode(mask_tokenizer.eos_token)[1]:                           
                continue

            ranked_ids_copy = copy.deepcopy(ranked_word_ids)
            
            logging.info(f"Token to be masked is {mask_tokenizer.decode(ranked_ids_copy[i])}")
            ranked_ids_copy[i] = mask_tokenizer.encode(mask_tokenizer.mask_token)[1]                                 
            sequence = restore_list(ranked_ids_copy, indices)
            
            masked_sentence = mask_tokenizer.decode(sequence)
            logging.info(f"Masked sentence is {masked_sentence}")

            input = mask_tokenizer.encode(masked_sentence, return_tensors='pt').to('cuda')
            mask_token_index = torch.where(input == mask_tokenizer.mask_token_id)[1]
            token_logits = mask_model(input).logits                                                                   
            mask_token_logits = token_logits[0, mask_token_index, :]                                   
            top_k_tokens = torch.topk(mask_token_logits, k=10, dim=1).indices[0].detach().cpu().tolist()[1:]

            top_k_token_words = mask_tokenizer.convert_ids_to_tokens(top_k_tokens)   
            
            for j in range(len(top_k_tokens)):
                logging.info(f"Substituted token is {top_k_token_words[j]}")
                ranked_ids_copy[i] = top_k_tokens[j]

                new_sequence = restore_list(ranked_ids_copy, indices)
                new_sentence = mask_tokenizer.decode(new_sequence)
                logging.info(f"New sentence generated is {new_sentence}")
            
                if leakage_criterion(sentence, new_sentence, cls_model, cls_tokenizer, args):  
                    logging.info("New sentence accepted") 
                    candidate_leakage_dicts.append({'candidate_leakage': new_sentence, 'leakage': sentence,
                                            'masked_leakage': masked_sentence, 'generated_token': top_k_token_words[j]})

    return candidate_leakage_dicts


def main(args):    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)                                    
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    
    mask_tokenizer = AutoTokenizer.from_pretrained(args.mask_tokenizer)
    mask_model = AutoModelForMaskedLM.from_pretrained(args.mask_model).to('cuda')
    cls_tokenizer = AutoTokenizer.from_pretrained(args.cls_tokenizer)
    cls_model = AutoModelForSequenceClassification.from_pretrained(args.cls_model).to('cuda')

    with open(args.input_file, 'r') as f:
        sentences = f.read().splitlines()

    candidate_leakage_dicts = generate_candidate_leakages(sentences, cls_model, mask_model, cls_tokenizer, mask_tokenizer, args)       

    with open(args.output_file, 'w') as f:
        data = {"args": vars(args), "generated_sentences": candidate_leakage_dicts}
        json.dump(data, f, indent=2)


if __name__ == "__main__":                                 
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, help="Path to input file containing seed leakages")
    parser.add_argument("--output_file", type=str, help="Path where to write generated sentences")
    parser.add_argument("--mask_tokenizer", type=str, default="roberta-large", help="Tokenizer for masked language model")
    parser.add_argument("--mask_model", type=str, default="roberta-large", help="Finetuned masked language model to use for word substitution")
    parser.add_argument("--cls_tokenizer", type=str, default="roberta-large", help="Tokenizer for classifer model")
    parser.add_argument("--cls_model", type=str, default="roberta-large", help="Finetuned classifer to find leakages")
    parser.add_argument("--acceptance_algo", type=str, default="classifier_score", help="Acceptance algorithm to accept a generated leakage")
    parser.add_argument("--word_rank_algo", type=str, default="word_attentions", help="Ranking algorithm to choose preferential words")
    parser.add_argument("--sentence_rank_algo", type=str, default=None, help="Ranking algorithm to choose preferential sentences")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)


""" 
mask_tokenizer and cls_tokenizer needs to be of the same model, else words maybe broken and tokenized differently by
the different tokenizers, and then word attentions or gradients don't have a meaning anymore
"""
