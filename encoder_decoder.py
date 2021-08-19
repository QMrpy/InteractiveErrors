import random
import torch
import json
import os
import warnings
import argparse
import logging
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, GPT2Tokenizer
from transformers import EncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


class PairedTextDataset(Dataset):
    def __init__(self, enc_tokenizer, dec_tokenizer, source_text, target_text, encoder_max_length=128, decoder_max_length=128):
        super().__init__()
        self.inputs = enc_tokenizer(source_text, padding="max_length", truncation=True, max_length=encoder_max_length)
        self.outputs = dec_tokenizer(target_text, padding="max_length", truncation=True, max_length=decoder_max_length)
        self.labels = self.outputs.input_ids.copy()

        self.labels = [
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(self.outputs.attention_mask, self.labels)]
        ]

        assert all([len(x) == encoder_max_length for x in self.inputs.input_ids])
        assert all([len(x) == decoder_max_length for x in self.outputs.input_ids])

    def __len__(self):
        return len(self.inputs.input_ids)

    def __getitem__(self, i):
        item = {
            "input_ids": torch.tensor(self.inputs.input_ids[i]), 
            "attention_mask": torch.tensor(self.inputs.attention_mask[i]),
            "decoder_input_ids": torch.tensor(self.outputs.input_ids[i]),
            "decoder_attention_mask": torch.tensor(self.outputs.attention_mask[i]),
            "labels": torch.tensor(self.labels[i])
            }

        return item


def train_lm(queries, train_dir, model_path, model, cls_tokenizer, gen_tokenizer, num_train_epochs=5, per_device_train_batch_size=2,
             per_device_eval_batch_size=2, warmup_steps=2000, weight_decay=0.01, learning_rate=5e-5, max_grad_norm=1.0, no_train=False):

    train_queries, val_queries = train_test_split(queries, test_size=0.2, random_state=42)
    train_dataset = PairedTextDataset(cls_tokenizer, gen_tokenizer, train_queries, train_queries)
    val_dataset = PairedTextDataset(cls_tokenizer, gen_tokenizer, val_queries, val_queries)

    metric = load_metric("rouge")

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(train_dir, 'results'),
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=os.path.join(train_dir, 'logs'),
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=10000,
        max_grad_norm=max_grad_norm,
        predict_with_generate=True,
        report_to=['tensorboard']
    )

    def compute_metrics(output):
        label_ids = output.label_ids
        pred_ids = output.predictions

        pred_str = gen_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = gen_tokenizer.eos_token_id
        label_str = gen_tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        rouge_output = metric.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset
    )

    if not no_train:
        trainer.train()
        trainer.save_model(model_path)

    return (model, cls_tokenizer, gen_tokenizer)


def generate_from_model(model, leakages, k=5):
    model, cls_tokenizer, gen_tokenizer = model

    candidate_leakage_dicts = []
    for leakage in tqdm(leakages):
        inputs = cls_tokenizer(leakage, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        outputs = model.generate(input_ids, attention_mask=attention_mask, num_return_sequences=k, top_p=0.95, top_k=20, do_sample=True)
        outputs = [gen_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        outputs = set(outputs)
        if leakage in outputs:
            outputs.remove(leakage)
        outputs = list(outputs)

        logging.info(f"{len(outputs)} sentences generated")
        for output in outputs:
            candidate_leakage_dicts.append({"leakage": leakage, "reconstructed_leakage": output})

    return candidate_leakage_dicts


def main(args):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.cls_model, args.gen_model)
    logging.info("Loaded encoder decoder model ")

    cls_tokenizer = BertTokenizer.from_pretrained(args.cls_model)
    cls_tokenizer.bos_token = cls_tokenizer.cls_token
    cls_tokenizer.eos_token = cls_tokenizer.sep_token

    GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    gen_tokenizer = GPT2Tokenizer.from_pretrained(args.gen_model)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token

    model.config.decoder_start_token_id = gen_tokenizer.bos_token_id
    model.config.eos_token_id = gen_tokenizer.eos_token_id
    model.config.max_length = 20
    model.config.min_length = 5
    model.config.no_repeat_ngram_size = 2
    model.early_stopping = True
    model.length_penalty = 2.0
    model.num_beams = 4

    with open(args.queries_file, "r") as f:
        queries = f.read().splitlines()

    logging.info("Will be fine tuning encoder decoder model")
    trained_model = train_lm(queries, args.train_dir, args.model_dir, model, cls_tokenizer, gen_tokenizer)

    with open(args.leakage_file, "r") as f:
        leakages = f.read().splitlines()

    logging.info("Reconstructing sentences from model")
    candidate_leakage_dict = generate_from_model(trained_model, leakages)

    with open(args.output_file, "w") as f:
        json_dict = {"args": vars(args), "candidate_leakages": candidate_leakage_dict}
        json.dump(json_dict, f, indent=2)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_file", type=str, help="Path of queries dataset")
    parser.add_argument("--leakage_file", type=str, help="Path of leakage dataset")
    parser.add_argument("--output_file", type=str, help="Path of output sentences file")
    parser.add_argument("--model_dir", type=str, help="Path where to save the fine tuned model to")
    parser.add_argument("--train_dir", type=str, help="Path where to save the train logs to")
    parser.add_argument("--cls_model", type=str, default="bert-large-uncased", help="Path or name of classifier model")
    parser.add_argument("--gen_model", type=str, default="gpt2-medium", help="Path or name of generator model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
