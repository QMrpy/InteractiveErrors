import argparse
import logging
import json
import os
import random
import copy
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


def convert_to_bart_mlm_format(lines):
    outputs = copy.deepcopy(lines)
    inputs = []

    for line in lines:
        span_length = np.random.poisson(lam=3)
        words = line.split()
        if len(words) == 0:
            continue
        start_id = random.choice(range(0, len(words)))

        masked_sentence = []
        masked = False
        for i in range(len(words)):
            if i < start_id or i >= start_id + span_length:
                masked_sentence.append(words[i])
            else:
                if not masked:
                    masked_sentence.append("<mask>")
                    masked = True

        if not masked:
            masked_sentence[-1] = "<mask>"

        masked_sentence = " ".join(masked_sentence)
        inputs.append(masked_sentence)

    return inputs, outputs


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, lines, truncate_length=128):
        inputs, outputs = convert_to_bart_mlm_format(lines)
        logging.info("Converted to BART Masked LM Format")
        
        self.input_ids = tokenizer(inputs, padding=True, truncation=True, max_length=truncate_length).input_ids
        self.labels = tokenizer(outputs, padding=True, truncation=True, max_length=truncate_length).input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        item = {"input_ids": torch.tensor(self.input_ids[i]), "labels": torch.tensor(self.labels[i])}
        return item


def train_lm(queries, output_dir, model_path, model_name="facebook/bart-large", num_train_epochs=5, per_device_train_batch_size=64,
             per_device_eval_batch_size=128, warmup_steps=1000, weight_decay=0.01, learning_rate=5e-5, max_grad_norm=1.0, truncate_length=128,
             no_train=False):

    tokenizer = BartTokenizer.from_pretrained(model_name)

    try:
        model = BartForConditionalGeneration.from_pretrained(model_path, forced_bos_token_id=tokenizer.bos_token_id)
        logging.info(f"Successfully loaded fine tuned model from path {model_path}")

    except OSError:
        model = BartForConditionalGeneration.from_pretrained(model_name, forced_bos_token_id=tokenizer.bos_token_id)
        pass
    
    train_queries, val_queries = train_test_split(queries, test_size=0.1, random_state=42)
    train_dataset = LineByLineTextDataset(tokenizer, train_queries, truncate_length)
    val_dataset = LineByLineTextDataset(tokenizer, val_queries, truncate_length)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(output_dir, 'results'),
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=20000,
        max_grad_norm=max_grad_norm,
        metric_for_best_model="eval_loss",
        report_to=['tensorboard']
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    if not no_train:
        logging.info("Fine tuning BART")
        trainer.train()
        trainer.save_model(model_path)

    return (model, tokenizer)


def generate_candidate_leakages(model, leakages, k=5):
    model, tokenizer = model
    masked_leakages, _ = convert_to_bart_mlm_format(leakages)
    candidate_leakage_dicts = []

    logging.info("Generating candidate leakages")
    for masked_leakage, leakage in tqdm(zip(masked_leakages, leakages)):
        input_ids = tokenizer(masked_leakage, return_tensors='pt').to('cuda')
        outputs = model.generate(input_ids['input_ids'], num_return_sequences=k, num_beams=k)
        output_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        output_sentences = set(output_sentences)
        if leakage in output_sentences:
            output_sentences.remove(leakage)
        output_sentences = list(output_sentences)

        for output in output_sentences:
            candidate_leakage_dicts.append({'candidate_leakage': output, 'leakage': leakage, 'masked_leakage': masked_leakage})
            
    return candidate_leakage_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,  help="Directory where to save output logs and sentences")
    parser.add_argument("--model_path", type=str, help="Path where to save fine tuned model and load fine tuned model from")
    parser.add_argument("--leakages_file", type=str, help="File with leakges")
    parser.add_argument("--queries_file", type=str, help="File with queries")
    parser.add_argument("--output_file", type=str, help="Path where to save generated leakages")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large", help="Model to use")
    parser.add_argument("--train_samples", type=int, default=10000, help="Number of queries to train on")
    parser.add_argument("--k", type=int, default=5, help="Number of candidate leakages to generate per leakage")
    parser.add_argument("--num_train_epochs", type=int, default=4, help="Number of epochs to train for")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Per device train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Per device eval batch size")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maxium gradient norm for clipping")
    parser.add_argument("--truncate_length", type=int, default=128, help="Maximum token length to truncate while training")
    parser.add_argument("--no_train", action="store_true", help="specify no training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    queries_file = args.queries_file
    with open(queries_file, 'r') as f:
        queries = f.read().splitlines()

    sampled_indices = np.random.choice(np.arange(len(queries)), size=min(args.train_samples, len(queries)), replace=False)
    queries = [queries[i] for i in sampled_indices] 

    model = train_lm(queries, args.output_dir, args.model_path, model_name=args.model_name,
                     num_train_epochs=args.num_train_epochs, per_device_train_batch_size=args.per_device_train_batch_size,
                     per_device_eval_batch_size=args.per_device_eval_batch_size, warmup_steps=args.warmup_steps,
                     weight_decay=args.weight_decay, learning_rate=args.learning_rate, max_grad_norm=args.max_grad_norm,
                     truncate_length=args.truncate_length, no_train=args.no_train)

    leakages_file = args.leakages_file
    with open(leakages_file, 'r') as f:
        leakages = f.read().splitlines()

    candidate_leakages_dict = generate_candidate_leakages(model, leakages, k=args.k)

    output_file = os.path.join(args.output_dir, args.output_file)
    with open(output_file, 'w') as file:
        data = {'args': vars(args), 'candidate_leakages': candidate_leakages_dict}
        json.dump(candidate_leakages_dict, file, indent=2)
