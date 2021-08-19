import json
import os
import logging
import argparse
import random

def main(args):
    input_src = args.input_dir

    if os.path.isfile(input_src):
        input_files = [input_src]

    else:
        input_files = os.listdir(input_src)
        input_files = [os.path.join(input_src, f) for f in input_files]
    logging.info(f'Will check generated queries for {len(input_files)} files.')

    gen_queries = []

    for file_name in input_files:
        logging.info(f"Checking {file_name}...")
        if file_name.endswith(".json"):

            with open(file_name, "r") as f:
                data = json.load(f)
                try:
                    data = data["output_sentences"]
                except:
                    try:
                        data = data["candidate_leakages"]
                    except:
                        pass
                    pass
                for i in data:
                    try:
                        gen_queries.append(i["generated_sentence"])
                    except:
                        gen_queries.append(i["candidate_leakage"])
                        pass

    gen_queries = list(set(gen_queries))
    logging.info(f"gen_queries has {len(gen_queries)} total generated sentences")

    gen_queries = random.sample(gen_queries, args.samples)
    logging.info(f"gen_queries has {len(gen_queries)} samples")

    with open(args.output_file, "w") as f:
        for i in gen_queries:
            f.write("1" + "\t" + i + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Input folder or file to process outputs from")
    parser.add_argument("--output_file", type=str, help="Processed results output file to be annotated")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to annotate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
