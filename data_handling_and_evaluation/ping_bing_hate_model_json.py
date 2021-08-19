"""
This script checks whether a query is hate or not using hosted bing model.
The input_src should should be a json file containing a dict of leakages and candidate_leakages, or a folder of them
It returns a file with scores of leakage and candidate leakage from the hosted Bing hate model
"""

import argparse
import logging
import os
import json
import requests

URL = "http://deeplearning.indexservedlmodelserve2-prod-co4.co4.ap.gbl:86/route/PeopleAlsoAskNews.HateV4PAA"


def check_hate(input_src, output_dir, url, data_prune, MAX_QUERIES_PER_REQUEST=30):
    if os.path.isfile(input_src):
        input_files = [input_src]
    else:
        input_files = os.listdir(input_src)
        input_files = [os.path.join(input_src, f) for f in input_files]
    logging.info(f'Will check queries for {input_files} files.')

    queries = set()

    for file_name in input_files:
        logging.info(f"Checking {file_name}...")
        if file_name.endswith(".json") == False:
            continue

        with open(file_name, 'r') as file:
            data = json.load(file)

        query_data = data["candidate_leakages"][:data_prune]
        for i in query_data:
            queries.add(i["leakage"])
            queries.add(i["candidate_leakage"])

        queries = list(queries)
        queries = [{"query": q} for q in queries]
        logging.info(f'Total number of queries to check {len(queries)}.')

        num_queries = len(queries)
        response = None
        results = []
        print(queries)

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

        results = [y for x in results for y in x]
        logging.info(f'Got hate scores for {len(results)} queries.')

        num_positive_queries = sum([r['hateScore'] > 0.5 for r in results])
        logging.info(f'Out of {len(results)}, {num_positive_queries} queries have been classified as hate.')

        output_file = os.path.join(output_dir + 'result_json_' + os.path.basename(file_name))              
        with open(output_file, 'w') as file:
            for result in results:
                file.write(result['query'] + '\t' + str(result['hateScore']) + '\n')
        logging.info(f'Wrote output to {output_file}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_src', type=str, help="Input file or directory")
    parser.add_argument('--output_dir', type=str, default='.', help="Path to output dir")
    parser.add_argument('--url', type=str, default=URL, help="URL to check for request")
    parser.add_argument('--data_prune', type=int, default=20000, help="Prune data to get results quickly")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    check_hate(args.input_src, args.output_dir, args.url, args.data_prune)
