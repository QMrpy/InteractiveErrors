import argparse
import json
import logging
import os
import warnings
import spacy
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments

from utils import read_file, grouper


def convert_to_t5_format(nlp, texts):
    """Takes in texts and converts it into t5 format."""

    inputs = []
    outputs = []
    original_texts = []

    for text, doc in zip(texts, nlp.pipe(texts, n_process=-1)):

        pairs = set()

        for chunk in doc.noun_chunks:
            if chunk.text == text:
                continue
            input_ = text[0 : chunk.start_char] + "<extra_id_0> " + text[chunk.end_char + 1 :]
            output = "<extra_id_0> " + chunk.text + " <extra_id_1> </s>"

            pairs.add((input_.strip(), output))

        for token in doc:
            left_edge_i = token.left_edge.i
            right_edge_i = token.right_edge.i
            chunk_length = right_edge_i - left_edge_i + 1
            if chunk_length / len(doc) > 0.5 or chunk_length > 10:  # if chunk is too long, just skip it
                continue

            input_ = str(doc[:left_edge_i]) + " <extra_id_0> " + str(doc[right_edge_i + 1 :])
            output = "<extra_id_0> " + str(doc[left_edge_i : right_edge_i + 1]) + " <extra_id_1> </s>"

            pairs.add((input_.strip(), output))

        for token in doc:
            if token.pos_ in ["NOUN", "PRON", "PROPN"]:  # we don't want to mask parts of noun chunks
                continue
            input_ = str(doc[: token.i]) + " <extra_id_0> " + str(doc[token.i + 1 :])
            output = "<extra_id_0> " + token.text + " <extra_id_1> </s>"

            pairs.add((input_.strip(), output))

        for (input_, output) in pairs:
            inputs.append(input_)
            outputs.append(output)
            original_texts.append(text)

    return inputs, outputs, original_texts


def convert_from_t5_format(input_, output):
    start_ix = output.find("<extra_id_0>")
    end_ix = output.find("<extra_id_1>")

    if start_ix != -1 and end_ix != -1:
        result = input_.replace("<extra_id_0>", output[start_ix + 12 : end_ix])
        result = " ".join(result.split())

        return result
    else:
        logging.debug(f"Couldn't convert from T5 format - input: {input_}, output: {output}.")

        return ""


class LineByLineTextDataset(Dataset):
    def __init__(self, nlp, tokenizer, lines, truncate_length=128):
        logging.info("Converting lines into t5 format...")
        inputs, outputs, _ = convert_to_t5_format(nlp, lines)
        logging.info("Finished converting into t5 format.")

        logging.info("Tokenizing the inputs...")
        self.input_ids = tokenizer(inputs, padding=True, truncation=True, max_length=truncate_length).input_ids
        logging.info("Tokenizing the outputs...")
        self.labels = tokenizer(outputs, padding=True, truncation=True, max_length=truncate_length).input_ids
        logging.info("Finished creating a dataset instance.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        item = {"input_ids": torch.tensor(self.input_ids[i]), "labels": torch.tensor(self.labels[i])}
        return item


def train_lm(
    generator_model_path,
    queries,
    output_dir,
    nlp,
    model_name="t5-base",
    no_training=False,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    learning_rate=5e-5,
    max_grad_norm=1.0,
    truncate_length=32,
    local_rank=-1,
):

    try:
        logging.info(f"Trying to load {model_name} from {generator_model_path} if it exists..")
        device = torch.device("cuda")
        model = T5ForConditionalGeneration.from_pretrained(generator_model_path).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        logging.info(f"Successfully loaded model and tokenizer from {generator_model_path}.")
        
        return (model, tokenizer)

    except OSError:
        logging.info("Model not found. Will need to train new model.")
        pass

    logging.info(f"Loading {model_name} model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    logging.info(f"{model_name} model loaded.")

    logging.info("Preparing training and validation datasets...")
    train_queries, val_queries = train_test_split(queries, test_size=0.1, random_state=42)
    train_dataset = LineByLineTextDataset(nlp, tokenizer, train_queries, truncate_length)
    val_dataset = LineByLineTextDataset(nlp, tokenizer, val_queries, truncate_length)
    logging.info("Training and validation datasets prepared.")

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "results"),
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        lr_scheduler_type="constant",
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        max_grad_norm=max_grad_norm,
        metric_for_best_model="eval_loss",
        report_to=["tensorboard"],
        local_rank=local_rank,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)

    if not no_training:
        logging.info("Training T5 model...")
        trainer.train()
        trainer.save_model(generator_model_path)
        logging.info(f"Saved {model_name} to {generator_model_path}")

    return (model, tokenizer)


def generate_candidate_leakages(
    model,
    leakages,
    nlp,
    per_device_eval_batch_size=16,
    truncate_length=64,
    k=3,
    max_length=20,
    temperature=1.0,
    use_tqdm=True,
):
    model, tokenizer = model
    candidate_leakage_dicts = []

    inputs, _, leakages = convert_to_t5_format(nlp, leakages)
    data = [{"input": input_, "leakage": leakage} for input_, leakage in zip(inputs, leakages)]

    model.to("cuda")
    model.eval()

    iterator = grouper(data, per_device_eval_batch_size)
    if use_tqdm:
        iterator = tqdm(iterator)

    for batch in iterator:
        batch = [x for x in batch if x is not None]
        batch_input = [x["input"] for x in batch]
        batch_input_ids = tokenizer(
            batch_input, padding=True, truncation=True, max_length=truncate_length, return_tensors="pt"
        ).input_ids.to("cuda")
        batch_outputs = (
            model.generate(
                batch_input_ids,
                min_length=5,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=k,
                num_beams=k,
                do_sample=True,
            )
            .detach()
            .cpu()
            .tolist()
        )

        batch = [x for x in batch for i in range(k)]
        for point, output in zip(batch, batch_outputs):
            tokens = tokenizer.convert_ids_to_tokens(output)
            tokens = [token for token in tokens if token not in ["<pad>", "<\s"]]

            output = tokenizer.convert_tokens_to_string(tokens)
            candidate_leakage = convert_from_t5_format(point["input"], output)
            candidate_leakage_dicts.append(
                {
                    "leakage": point["leakage"],
                    "candidate_leakage": candidate_leakage,
                    "masked_leakage": point["input"],
                    "output": output,
                }
            )

    return candidate_leakage_dicts


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Directory where write generated candidate leakages")
    parser.add_argument("--gen_model_path", type=str, default="t5-base")
    parser.add_argument("--leakages_file", type=str, help="File with leakages")
    parser.add_argument("--queries_file", type=str, help="File with queries queries")
    parser.add_argument("--output_file", type=str, help="Filename for generated leakages")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Model to use")
    parser.add_argument("--k", type=int, default=5, help="Number of candidate leakages to generate per leakage")
    parser.add_argument("--max_length_generation", type=int, default=20, help="Maximum token length while generating.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature while generating.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Per device train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Per device eval batch size")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maxium gradient norm for clipping")
    parser.add_argument("--truncate_length", type=int, default=32, help="Maximum token length to truncate while training")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_lg", help="Spacy model to use")
    parser.add_argument("--no_training", action="store_true", help="If not to train the T5 model")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    os.environ["TOKENIZER_PARALLELISM"] = "false"

    queries_file = args.queries_file
    logging.info(f"Reading queries from {queries_file}...")
    queries = read_file(queries_file)
    logging.info(f"{len(queries)} queries read.")

    generator_model_path = args.gen_model_path
    
    nlp = spacy.load(args.spacy_model)

    model = train_lm(
        generator_model_path,
        queries,
        args.output_dir,
        nlp,
        model_name=args.model_name,
        no_training=args.no_training,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        truncate_length=args.truncate_length,
        local_rank=args.local_rank,
    )
    logging.info("T5 model trained or loaded")

    leakages_file = args.leakages_file
    logging.info(f"Reading leakages from {leakages_file}...")
    leakages = read_file(leakages_file)
    logging.info(f"{len(leakages)} leakages read.")

    logging.info("Generating candidate leakages...")
    candidate_leakage_dicts = generate_candidate_leakages(
        model,
        leakages,
        nlp,
        args.per_device_eval_batch_size,
        args.truncate_length,
        k=args.k,
        max_length=args.max_length_generation,
        temperature=args.temperature,
    )
    logging.info(f"{len(candidate_leakage_dicts)} leakages generated.")

    output_file = os.path.join(args.output_dir, args.output_file)
    logging.info(f"Writing leakages to {output_file}...")
    with open(output_file, "w") as file:
        data = {"args": vars(args), "candidate_leakages": candidate_leakage_dicts}
        json.dump(data, file, indent=2)
