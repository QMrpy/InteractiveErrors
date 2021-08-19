import argparse
import json
import logging
import math
import pandas
import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm


def get_hate_scores(candidate_leakages, url, MAX_QUERIES_PER_REQUEST=30):

    queries = [x['candidate_leakage'] for x in candidate_leakages]
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


def compute_semantic_similarity(candidate_leakages, sentence_embedding_model):
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
    for point in tqdm(candidate_leakages):
        similarities.append(_helper(point['leakage'], point['candidate_leakage']))
    
    return similarities


def compute_bleu_scores(candidate_leakages):
    bleu_scores = []
    chencherry = SmoothingFunction()

    for point in tqdm(candidate_leakages):
        reference = point['leakage'].split()  
        hypothesis = point['candidate_leakage'].split()  
        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=chencherry.method2)
        bleu_scores.append(bleu_score)

    return bleu_scores


def compute_self_bleu_score(candidate_leakages):
    leakage_candidate_map = {}

    for point in candidate_leakages:
        leakage = point['leakage']
        candidate_leakage = point['candidate_leakage']

        if leakage not in leakage_candidate_map:
            leakage_candidate_map[leakage] = set()
        leakage_candidate_map[leakage].add(candidate_leakage)

    self_bleu_scores = []
    chencherry = SmoothingFunction()

    for candidate_leakages_group in tqdm(leakage_candidate_map.values()):
        if len(candidate_leakages_group) == 1:
            continue

        scores = []
        for candidate_leakage in candidate_leakages_group:
            references = list(candidate_leakages_group)
            references.remove(candidate_leakage)

            references = [x.split() for x in references]
            bleu_score = sentence_bleu(references, candidate_leakage.split(),
                                       smoothing_function=chencherry.method2)
            scores.append(bleu_score)

        self_bleu_scores.append(mean(scores))

    return self_bleu_scores


def compute_lm_scores(candidate_leakages, lm_path):
    logging.info(f"Loading language model from {lm_path}..")
    model = GPT2LMHeadModel.from_pretrained(lm_path).to('cuda')
    tokenizer = GPT2TokenizerFast.from_pretrained(lm_path)
    logging.info("Language model loaded.")

    log_likelihoods = []
    lengths = []

    for point in tqdm(candidate_leakages):
        encoding = tokenizer(point['candidate_leakage'], return_tensors='pt')
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


def filter_candidate_leakages(args, candidate_leakages):
    pairs_seen = set()
    filtered_candidate_leakages = []

    if args.input_file_tsv:
        for index, row in candidate_leakages.iterrows():
            sents = row['sentences'][1:-1].split(",")
            sents = [sent.strip() for sent in sents]
            sents = [sent[1:-1] for sent in sents]
            for sent in sents:
                pair = (row['original'], sent)
                if sent == '' or pair in pairs_seen:
                    continue
                pairs_seen.add(pair)
                point = {"leakage": row['original'], "score": row['score'], 'candidate_leakage': sent}
                filtered_candidate_leakages.append(point)
    else:
        for point in candidate_leakages:
            pair = (point['leakage'], point['candidate_leakage'])
            if point['candidate_leakage'] == '' or pair in pairs_seen:
                continue
            pairs_seen.add(pair)
            filtered_candidate_leakages.append(point)

    return filtered_candidate_leakages


def mean(array):
    if len(array) == 0:
        return float('nan')

    return sum(array) / len(array)


def main(args):
    if args.input_file_tsv:
        df = pandas.read_csv(args.input_file, sep='\t')
        df1 = df.loc[df['score']<=args.score_up]
        candidate_leakages = df1.loc[df1['score']>=args.score_lo]
    else:
        with open(args.input_file, 'r') as file:
            data = json.load(file)
        try:
            candidate_leakages = data['candidate_leakages']
        except:
            candidate_leakages = data
            pass

    logging.info(f'Filtering {len(candidate_leakages)} candidate leakages..')
    candidate_leakages = filter_candidate_leakages(args, candidate_leakages)
    logging.info(f'Filtered to {len(candidate_leakages)} candidate_leakages.')

    unique_candidate_leakages = len(set([x['candidate_leakage'] for x in candidate_leakages]))
    evaluation_meta = {'unique_candidate_leakages': unique_candidate_leakages}

    logging.info("Computing bleu scores..")
    bleu_scores = compute_bleu_scores(candidate_leakages)
    evaluation_meta['bleu_score'] = mean(bleu_scores)

    logging.info("Computing self bleu scores..")
    self_bleu_scores = compute_self_bleu_score(candidate_leakages)
    evaluation_meta['self_bleu_score'] = mean(self_bleu_scores)

    if args.lm_path:
        logging.info("Calculating lm scores..")
        log_likelihoods, perplexity = compute_lm_scores(candidate_leakages, args.lm_path)
        evaluation_meta['perplexity'] = perplexity

    if args.sentence_embedding_model:
        logging.info("Calculating semantic similarity scores..")
        semantic_similaries = compute_semantic_similarity(candidate_leakages, args.sentence_embedding_model)
        evaluation_meta['semantic_similaries'] = mean(semantic_similaries)

    print(evaluation_meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="file to evaluate")
    parser.add_argument("--input_file_tsv", action="store_true", help="File type is a tsv. original-template-score-stddev-sentences")
    parser.add_argument("--lm_path", type=str, help="lm path")
    parser.add_argument("--score_up", type=float, default=0.98, help="score upper limit")
    parser.add_argument("--score_lo", type=float, default=0.7, help="score lower limit")
    parser.add_argument("--sentence_embedding_model", type=str, default="sentence-transformers/stsb-roberta-large", help="model to get sentence embedding")
    parser.add_argument("--url", default="http://deeplearning.indexservedlmodelserve2-prod-co4.co4.ap.gbl:86/route/PeopleAlsoAskNews.HateV4PAA", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    nltk_logger = logging.getLogger('nltk')
    nltk_logger.setLevel(logging.ERROR)

    main(args)
