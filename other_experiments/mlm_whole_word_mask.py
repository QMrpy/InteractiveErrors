import argparse
import json
import os
import random
import copy
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from tqdm import tqdm


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, lines, truncate_length=64):
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=truncate_length)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = {"input_ids": torch.tensor(self.examples[i], dtype=torch.long)}
        return example


def train_lm(queries, output_dir, model_path, model_name="bert-large-cased-whole-word-masking", num_train_epochs=5, per_device_train_batch_size=64,
             per_device_eval_batch_size=128, warmup_steps=500, weight_decay=0.01, learning_rate=5e-5, max_grad_norm=1.0, truncate_length=64,
             no_train=False):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    train_queries, val_queries = train_test_split(queries, test_size=0.1, random_state=42)
    train_dataset = LineByLineTextDataset(tokenizer, train_queries, truncate_length)
    val_dataset = LineByLineTextDataset(tokenizer, val_queries, truncate_length)

    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'results'),
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        max_grad_norm=max_grad_norm,
        metric_for_best_model="eval_loss",
        report_to=['tensorboard']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    if not no_train:
        trainer.train()
        trainer.save_model(model_path)

    return (model, tokenizer)


def generate_candidate_leakages(model, leakages, k=5):
    model, tokenizer = model
    unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0, top_k=k)
    candidate_leakage_dicts = []

    for leakage in tqdm(leakages):
        leakage_copy = copy.deepcopy(leakage)
        words = leakage_copy.split()
        random_ix = random.choice(range(0, len(words)))
        words[random_ix] = "[MASK]"
        masked_leakage = " ".join(words)

        outputs = unmasker(masked_leakage)
        for output in outputs:
            candidate_leakage_dicts.append({'candidate_leakage': output["sequence"], 'leakage': leakage,
                                            'masked_leakage': masked_leakage, 'generated_token': output["token_str"]})

    return candidate_leakage_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="Directory with input data")
    parser.add_argument("--output_dir", type=str,help="Directory where to save output results")
    parser.add_argument("--leakages_file", type=str, help="File with leakages")
    parser.add_argument("--queries_file", type=str, help="File with queries")
    parser.add_argument("--output_file", type=str, help="File where to save generated leakages")
    parser.add_argument("--model_name", type=str, default="bert-large-cased-whole-word-masking", help="Model to use")
    parser.add_argument("--model_path", type=str, help="Full path of where to save fine tuned model")
    parser.add_argument("--k", type=int, default=5, help="Number of candidate leakages to generate per leakage")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Per device train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Per device eval batch size")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maxium gradient norm for clipping")
    parser.add_argument("--truncate_length", type=int, default=64, help="Maximum token length to truncate while training")
    parser.add_argument("--no_train", action="store_true", help="specify no training")

    args = parser.parse_args()

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    queries_file = os.path.join(args.data_dir, args.queries_file)
    with open(queries_file, 'r') as f:
        queries = f.read().splitlines()

    model = train_lm(queries=queries, output_dir=args.output_dir, model_path=args.model_path, model_name=args.model_name,
                     num_train_epochs=args.num_train_epochs, per_device_train_batch_size=args.per_device_train_batch_size,
                     per_device_eval_batch_size=args.per_device_eval_batch_size, warmup_steps=args.warmup_steps,
                     weight_decay=args.weight_decay, learning_rate=args.learning_rate, max_grad_norm=args.max_grad_norm,
                     truncate_length=args.truncate_length, no_train=args.no_train)

    leakages_file = os.path.join(args.data_dir, args.leakages_file)
    with open(leakages_file, 'r') as f:
        leakages = f.read().splitlines()

    candidate_leakages_dict = generate_candidate_leakages(model, leakages, k=args.k)

    output_file = os.path.join(args.output_dir, args.output_file)
    with open(output_file, 'w') as file:
        data = {'args': args, 'candidate_leakages': candidate_leakages_dict}
        json.dump(candidate_leakages_dict, file, indent=2)
