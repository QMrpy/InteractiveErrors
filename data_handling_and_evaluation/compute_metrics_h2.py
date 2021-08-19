import argparse
import json
import logging
import math
import os
import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm


def get_hate_scores(generated_sentences, url, MAX_QUERIES_PER_REQUEST=30):

    queries = [x['generated_sentence'] for x in generated_sentences]
    num_queries = len(queries)
    results = []

    for i in range(0, num_queries, MAX_QUERIES_PER_REQUEST):
        request_json = queries[i: i + MAX_QUERIES_PER_REQUEST]
        logging.debug(f'Sending a POST request with {len(request_json)} queries to {url}..')
        response = requests.post(url, json=request_json)
        logging.debug(f'Server responded with status code: {response.status_code}.')
        if response.status_code != 200:
            continue

        result = response.json()['result']
        results.append(result)
        logging.debug(f'Response contains hate scores for {len(result)} queries.')

    hate_scores = [y for x in results for y in x]
    logging.info(f'Got hate scores for {len(results)} queries.')
    
    return hate_scores


def compute_semantic_similarity(generated_sentences, sentence_embedding_model):
    logging.info(f'Loading sentence embedding model {sentence_embedding_model}..')
    tokenizer = AutoTokenizer.from_pretrained(sentence_embedding_model)
    model = AutoModel.from_pretrained(sentence_embedding_model).to('cuda')
    logging.info(f'Sentence embedding model {sentence_embedding_model} loaded.')

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def _helper(text1, text2):
        input_ids1 = tokenizer([text1], truncation=True, max_length=128, return_tensors='pt').input_ids.to('cuda')
        input_ids2 = tokenizer([text2], truncation=True, max_length=128, return_tensors='pt').input_ids.to('cuda')

        with torch.no_grad():
            embedding1 = model(input_ids1)[0].mean(dim=1)
            embedding2 = model(input_ids2)[0].mean(dim=1)
            return cos(embedding1, embedding2).item()

    similarities = []
    for point in tqdm(generated_sentences):
        similarities.append(_helper(point['selected_sentence'], point['generated_sentence']))
    
    return similarities


def compute_bleu_scores(generated_sentences):
    bleu_scores = []
    chencherry = SmoothingFunction()

    for point in tqdm(generated_sentences):
        reference = point['selected_sentence'].split()  
        hypothesis = point['generated_sentence'].split()  
        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=chencherry.method2)
        bleu_scores.append(bleu_score)

    return bleu_scores


def compute_self_bleu_score(generated_sentences):
    leakage_candidate_map = {}

    for point in generated_sentences:
        leakage = point['selected_sentence']
        candidate_leakage = point['generated_sentence']

        if leakage not in leakage_candidate_map:
            leakage_candidate_map[leakage] = set()
        leakage_candidate_map[leakage].add(candidate_leakage)

    self_bleu_scores = []
    chencherry = SmoothingFunction()

    for generated_sentences_group in tqdm(leakage_candidate_map.values()):
        if len(generated_sentences_group) == 1:
            continue

        scores = []
        for candidate_leakage in generated_sentences_group:
            references = list(generated_sentences_group)
            references.remove(candidate_leakage)

            references = [x.split() for x in references]  
            bleu_score = sentence_bleu(references, candidate_leakage.split(),
                                       smoothing_function=chencherry.method2)
            scores.append(bleu_score)

        self_bleu_scores.append(mean(scores))

    return self_bleu_scores


def compute_lm_scores(generated_sentences, lm_path):
    logging.info(f"Loading language model from {lm_path}..")
    model = GPT2LMHeadModel.from_pretrained(lm_path).to('cuda')
    tokenizer = GPT2TokenizerFast.from_pretrained(lm_path)
    logging.info("Language model loaded.")

    log_likelihoods = []
    lengths = []

    for point in tqdm(generated_sentences):
        encoding = tokenizer(point['generated_sentence'], return_tensors='pt')
        input_ids = encoding.input_ids.to('cuda')
        target_ids = input_ids.clone()

        with torch.no_grad():
            log_likelihood = model(input_ids, labels=target_ids)[0]
            
        if math.isnan(log_likelihood.item()):
            log_likelihoods.append(0)
            lengths.append(0)
            continue

        log_likelihoods.append(log_likelihood.item())
        lengths.append(input_ids.shape[-1])

    average_ll = sum([ll * l for ll, l in zip(log_likelihoods, lengths)]) / sum(lengths)
    perplexity = math.exp(average_ll)

    return log_likelihoods, perplexity


def filter_generated_sentences(generated_sentences):
    pairs_seen = set()
    filtered_generated_sentences = []

    for point in generated_sentences:
        pair = (point['selected_sentence'], point['generated_sentence'])
        if point['generated_sentence'] == '' or pair in pairs_seen:
            continue
        pairs_seen.add(pair)
        filtered_generated_sentences.append(point)

    return filtered_generated_sentences


def mean(array):
    if len(array) == 0:
        return float('nan')

    return sum(array) / len(array)


def main(args):
    input_src = args.input_data

    if os.path.isfile(input_src):
        input_files = [input_src]

    else:
        input_files = os.listdir(input_src)
        input_files = [os.path.join(input_src, f) for f in input_files]
    logging.info(f'Will check generated queries for {len(input_files)} files.')

    generated_sentences = []

    for file_name in input_files:
        logging.info(f"Checking {file_name}...")
        if file_name.endswith(".json"):

            with open(file_name, "r") as f:
                data = json.load(f)
                data = data["output_sentences"]

                generated_sentences.extend(data)

    logging.info(f'Filtering {len(generated_sentences)} candidate leakages..')
    generated_sentences = filter_generated_sentences(generated_sentences)
    logging.info(f'Filtered to {len(generated_sentences)} generated_sentences.')

    unique_generated_sentences = len(set([x['generated_sentence'] for x in generated_sentences]))
    evaluation_meta = {'unique_generated_sentences': unique_generated_sentences}

    logging.info("Computing bleu scores..")
    bleu_scores = compute_bleu_scores(generated_sentences)
    evaluation_meta['bleu_score'] = mean(bleu_scores)

    logging.info("Computing self bleu scores..")
    self_bleu_scores = compute_self_bleu_score(generated_sentences)
    evaluation_meta['self_bleu_score'] = mean(self_bleu_scores)

    if args.lm_path:
        logging.info("Calculating lm scores..")
        log_likelihoods, perplexity = compute_lm_scores(generated_sentences, args.lm_path)
        evaluation_meta['perplexity'] = perplexity

    if args.sentence_embedding_model:
        logging.info("Calculating semantic similarity scores..")
        semantic_similaries = compute_semantic_similarity(generated_sentences, args.sentence_embedding_model)
        evaluation_meta['semantic_similaries'] = mean(semantic_similaries)

    print(evaluation_meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="file or directory of json files to evaluate")
    parser.add_argument("--lm_path", type=str, help="lm path")
    parser.add_argument("--sentence_embedding_model", type=str, default="sentence-transformers/stsb-roberta-large", help="model to get sentence embedding")
    parser.add_argument("--url", default="http://deeplearning.indexservedlmodelserve2-prod-co4.co4.ap.gbl:86/route/PeopleAlsoAskNews.HateV4PAA", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    nltk_logger = logging.getLogger('nltk')
    nltk_logger.setLevel(logging.ERROR)

    main(args)
