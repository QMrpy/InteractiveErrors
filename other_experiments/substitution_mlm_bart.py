import argparse
import os
import random
import logging
import copy
import json
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartTokenizer, BartForConditionalGeneration


def subword_aggregation(sentence, attentions, tokenizer):
    attentions = attentions[1:-1]
    tokens = tokenizer.tokenize(sentence)
    
    word_attentions = []
    word_idx = []
    cur_word_pos = 0
    word_attention = 0

    for i in range(len(tokens)):
        if i == 0:
            word_attention += attentions[i]
            word_idx.append(cur_word_pos)

        elif i != 0 and not tokens[i].startswith("Ġ"):
            word_attention += attentions[i]

        else:
            word_attentions.append(word_attention)
            cur_word_pos += 1
            word_attention = 0
            word_attention += attentions[i]
            word_idx.append(cur_word_pos)

    word_attentions.append(word_attention)      
    assert cur_word_pos == len(word_attentions) - 1, "Word attentions haven't been properly computed"
    
    return word_attentions, word_idx


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

        word_attentions, word_idx = subword_aggregation(sentence, attentions, tokenizer)   
        
        rank_ids = sorted([(word_attentions[i], word_idx[i]) for i in range(len(word_idx))], reverse=True)                     
        rank_ids = [r[1] for r in rank_ids]
        
        return rank_ids, word_attentions
    
    elif args.word_rank_algo == "word_gradients":
        logging.warning("Will use subwords for ranking. Not recommended to proceed")

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
    gen_sentence_dicts = []

    try:
        sentences = rank_sentences(sentences, args)

        for sentence in sentences:
            logging.info(f"Current sentence is: {sentence}")
            ranked_word_ids, word_attention_scores = rank_words(sentence, cls_model, cls_tokenizer, args)

            for i in range(len(ranked_word_ids)):
                sentence_copy = copy.deepcopy(sentence)
                sentence_copy = sentence_copy.split()

                if ranked_word_ids[i] >= len(sentence_copy):
                    logging.warn("There is some error in ranked_word_ids length")
                    continue

                masked_token = sentence_copy[ranked_word_ids[i]]
                logging.info(f"Token to be masked is: {masked_token}")
                sentence_copy[ranked_word_ids[i]] = "<mask>"
                
                masked_sentence = copy.deepcopy(sentence_copy)
                sentence_copy = " ".join(sentence_copy)
                logging.info(f"Masked sentence is: {sentence_copy}")
                
                input_ids = mask_tokenizer(sentence_copy, return_tensors='pt', padding=True, truncation=True, max_length=128).input_ids.to('cuda')
                outputs = mask_model.generate(input_ids, min_length=5, max_length=20, num_return_sequences=5, top_k=10, top_p=0.95, do_sample=True).detach().cpu().tolist()
                gen_sentences = [mask_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

                gen_sentences = set(gen_sentences)
                if sentence in gen_sentences:
                    gen_sentences.remove(sentence)
                gen_sentences = list(gen_sentences)
                logging.info(f"{len(gen_sentences)} sentences generated")

                for j in range(len(gen_sentences)):
                    logging.info(f"Generated sentence {j} is: {gen_sentences[j]}")

                for j in gen_sentences:
                    gen_sentence_dicts.append({'generated_sentence': j, 'leakage': sentence, 'word_attentions': word_attention_scores, 'masked_leakage': sentence_copy, 'masked_token': masked_token})
                    
                    if leakage_criterion(sentence, j, cls_model, cls_tokenizer, args):
                        logging.info(f"Generated sentence accepted: {j}")
                        candidate_leakage_dicts.append({'candidate_leakage': j, 'leakage': sentence, 'word_attentions': word_attention_scores, 'masked_leakage': sentence_copy, 'masked_token': masked_token})

    finally:
        logging.warning("Code encountered some error or was interrupted. Exiting and saving outputs")

        with open(args.gen_output_file, 'w') as f:
            data = {"args": vars(args), "generated_sentences": gen_sentence_dicts}
            json.dump(data, f, indent=2)

        with open(args.output_file, 'w') as f:
            data = {"args": vars(args), "candidate_leakages": candidate_leakage_dicts}
            json.dump(data, f, indent=2)

    return candidate_leakage_dicts, gen_sentence_dicts


def main(args):    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)                                    
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    
    mask_tokenizer = BartTokenizer.from_pretrained(args.mask_tokenizer)
    mask_model = BartForConditionalGeneration.from_pretrained(args.mask_model).to('cuda')
    cls_tokenizer = AutoTokenizer.from_pretrained(args.cls_tokenizer)
    cls_model = AutoModelForSequenceClassification.from_pretrained(args.cls_model).to('cuda')

    with open(args.input_file, 'r') as f:
        sentences = f.read().splitlines()

    candidate_leakage_dicts, generated_sentence_dicts = generate_candidate_leakages(sentences, cls_model, mask_model, cls_tokenizer, mask_tokenizer, args)       

    with open(args.gen_output_file, 'w') as f:
        data = {"args": vars(args), "generated_sentences": generated_sentence_dicts}
        json.dump(data, f, indent=2)

    with open(args.output_file, 'w') as f:
        data = {"args": vars(args), "candidate_leakages": candidate_leakage_dicts}
        json.dump(data, f, indent=2)


if __name__ == "__main__":                                 
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, help="Path to input file containing seed leakages")
    parser.add_argument("--output_file", type=str, help="Path where to write generated candidate leakages")
    parser.add_argument("--gen_output_file", type=str, help="Path where to write entire set of generated sentences")
    parser.add_argument("--mask_tokenizer", type=str, default="facebook/bart-large", help="Tokenizer for masked language model")
    parser.add_argument("--mask_model", type=str, default="facebook/bart-large", help="Finetuned masked language model to use for word substitution")
    parser.add_argument("--cls_tokenizer", type=str, default="roberta-large", help="Tokenizer for classifer model")
    parser.add_argument("--cls_model", type=str, default="roberta-large", help="Finetuned classifer to find leakages")
    parser.add_argument("--acceptance_algo", type=str, default="classifier_score", help="Acceptance algorithm to accept a generated leakage")
    parser.add_argument("--word_rank_algo", type=str, default="word_attentions", help="Ranking algorithm to choose preferential words")
    parser.add_argument("--sentence_rank_algo", type=str, default=None, help="Ranking algorithm to choose preferential sentences")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

