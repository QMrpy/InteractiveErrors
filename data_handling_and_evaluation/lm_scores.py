import torch
import argparse
import logging
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, BartForConditionalGeneration, T5ForConditionalGeneration


def main(args):
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    with open(args.input_file, "r") as f:
        data = f.read().splitlines()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.model_name == "gpt2-medium":
        model = GPT2LMHeadModel.from_pretrained(args.model_path).to("cuda")
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model_name == "facebook/bart-large":
        model = BartForConditionalGeneration.from_pretrained(args.model_path).to("cuda")
    elif args.model_name == "t5-base":
        model = T5ForConditionalGeneration.from_pretrained(args.model_path).to("cuda")
    else:
        raise ValueError("Could not find the specified model type")

    lm_scores = []
    expected_score = 0
    for i in tqdm(data):
        class_label, sentence = i.split(sep="\t")

        input_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True, max_length=128).input_ids.to("cuda")

        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=input_ids).loss.cpu().numpy()
            
        if np.isnan(loss):
            lm_scores.append(0.0)
            continue
        
        lm_scores.append(np.exp(-loss))
        expected_score += np.exp(-loss) 

    logging.info(f"Expected LM scores for the dataset = {expected_score / len(data)}")

    with open(args.output_file, "w") as f:
        for i in range(len(data)):
            class_label, sentence = data[i].split(sep="\t")
            f.write(sentence + "\t" + str(class_label) + "\t" + str(lm_scores[i]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to input file to evaluate")
    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Base name of the language model to find scores")
    parser.add_argument("--model_path", type=str, help="Path to fine tuned language model")
    parser.add_argument("--output_file", type=str, help="Output file with scores and class labels")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)    