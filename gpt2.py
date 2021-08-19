import argparse
import json
import logging
import os

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm


def generate_candidate_leakages(leakage, model, tokenizer, args):
    device = "cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda"

    candidate_leakage_dicts = []

    input_split = leakage.split()
    length_input = len(input_split)

    for i in range(args.min_prefix_length, min(args.max_prefix_length, length_input - 1)):
        input_context = " ".join(input_split[:i])
        input_ids = tokenizer(input_context, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids=input_ids, min_length=5, max_length=20, do_sample=True, top_k=20, top_p=0.97, num_return_sequences=5)
        
        for tokenized_text in outputs:
            candidate_leakage = tokenizer.decode(tokenized_text.tolist(), skip_special_tokens=True)
            candidate_leakage_dicts.append({'leakage': leakage, 'prefix': input_context, 'candidate_leakage': candidate_leakage})

    return candidate_leakage_dicts


def main(args):
    device = "cuda"

    logging.info(f"Loading GPT2 model and tokenizer from {args.pretrained_model}..")
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.to(device)
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained(args.pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    logging.info("GPT2 model and tokenizer loaded.")

    input_fp = args.leakages_file
    output_fp = args.output_file

    logging.info(f"Reading leakages from {input_fp} and generating new leakages..")
    candidate_leakage_dicts = []

    with open(input_fp, 'r') as input_file:
        leakages = input_file.readlines()
        for leakage in tqdm(leakages):
            candidate_leakage_dicts.extend(generate_candidate_leakages(leakage.strip(), model, tokenizer, args))
    logging.info(f"{len(candidate_leakage_dicts)} candidate leakages generated.")

    with open(output_fp, 'w') as file:
        data = {'args': vars(args), 'candidate_leakages': candidate_leakage_dicts}
        json.dump(data, file, indent=2)
    logging.info(f"Candidate leakages written to {output_fp}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leakages_file", type=str, help="File with leakges")
    parser.add_argument("--output_file", type=str, help="Path where to save generated sentences")
    parser.add_argument("--pretrained_model", type=str, default="gpt2-medium")
    parser.add_argument("--min_prefix_length", type=int, default=2)
    parser.add_argument("--max_prefix_length", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    logging_level = logging.INFO if not args.quiet else logging.ERROR
    logging.basicConfig(level=logging_level)

    main(args)
