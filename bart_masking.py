import re
import json
import spacy
import torch
import random
import logging
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import BartTokenizer, BartForConditionalGeneration


def remove_adjacent_masks(text):
	regex = r'\b(\w+)(?:\W+\1\b)+' 

	return re.sub(regex, r'\1', text, flags=re.IGNORECASE)


def mask_pos_tags(args, sentence):
	nlp = spacy.load(args.spacy_model)
	pos_tags = ["VERB", "ADJ", "ADV", "NOUN", "PROPN"]

	tokens = nlp(sentence)
	masks = ["".join(seq) for seq in itertools.product("01", repeat=min(12, len(tokens)))]
	masked_sentences = set()

	for mask in masks:
		masked_sentence = []

		for i in range(len(tokens)):
			if tokens[i].pos_ in pos_tags and mask[i] == "1":
				masked_sentence.append("<mask>")

			else:
				masked_sentence.append(tokens[i].text)
		
		masked_sentence = ' '.join(masked_sentence)
		masked_sentence = remove_adjacent_masks(masked_sentence)
		masked_sentences.add(masked_sentence)

	masked_sentences = list(masked_sentences)

	return masked_sentences


def main(args):
	RANDOM_SEED = 42
	random.seed(RANDOM_SEED)
	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	torch.cuda.manual_seed(RANDOM_SEED)

	tokenizer = BartTokenizer.from_pretrained(args.model_name)

	try:
		gen_model = BartForConditionalGeneration.from_pretrained(args.model_path, forced_bos_token_id=tokenizer.bos_token_id).to("cuda")
		logging.info(f"Successfully loaded fine tuned model from path {args.model_path}")

	except OSError:
		gen_model = BartForConditionalGeneration.from_pretrained(args.model_name, forced_bos_token_id=tokenizer.bos_token_id).to("cuda")
		logging.info("Could not load fine tuned model from path. Using non-fine tuned model")
		pass

	sim_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

	with open(args.input_file, "r") as f:
		data = f.read().splitlines()

	scored_sentences = []
	for sentence in tqdm(data):
		masked_templates = []

		logging.info(f"Generating mask templates for sentence: {sentence}")
		masked_sentences = mask_pos_tags(args, sentence)
		sentence_embedding = sim_model.encode([sentence], convert_to_tensor=True)

		for masked_sentence in masked_sentences:
			logging.info(f"Generating from masked template sentence: {masked_sentence}")
			input_ids = tokenizer(masked_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids.to("cuda")

			outputs = gen_model.generate(input_ids, num_return_sequences=10, top_k=50, top_p=0.99, do_sample=True)
			output_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
			output_sentences = list(set(output_sentences))
			logging.info(f"Generated {len(output_sentences)}")

			for i in range(len(output_sentences)):
				logging.info(f"Generated sentence {i} is: {output_sentences[i]}")

			gen_sentence_embeddings = sim_model.encode(output_sentences, convert_to_tensor=True)

			similarity_cosine_scores = util.pytorch_cos_sim(sentence_embedding, gen_sentence_embeddings)
			
			mask_score = torch.mean(similarity_cosine_scores).item()
			logging.info(f"Score of masked template based on sentence similarities is {mask_score}")

			mask_std = torch.std(similarity_cosine_scores).item()

			masked_templates.append({"masked_sentence": masked_sentence, "mask_score": mask_score, "mask_standard_deviation": mask_std, "generated_sentences": output_sentences})

		masked_templates = sorted(masked_templates, key=lambda x: x["mask_score"], reverse=True)
		scored_sentences.append({"sentence": sentence, "masked_templates": masked_templates})

	with open(args.output_file, "w") as f:
		dump = {"args": vars(args), "scored_sentences": scored_sentences}
		json.dump(dump, f, indent=2)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_file", type=str, help="Full path of input file containing leakages")
	parser.add_argument("--output_file", type=str, help="Full path of output file with scores, masked templates and generated sentences")
	parser.add_argument("--spacy_model", type=str, default="en_core_web_lg", help="Spacy model corresponding to the language used. Default is English")
	parser.add_argument("--model_name", type=str, default="facebook/bart-large", help="Generator model name to use")
	parser.add_argument("--model_path", type=str, help="Full path of fine tuned BART model to use")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)

	main(args)
