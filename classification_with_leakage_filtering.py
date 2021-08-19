import logging
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from utils import TextDataset, compute_metrics, move_dict_to_device, separate_labels


def train_classification_model(args, train_data, val_data):
    """Trains classification model."""

    classification_model_path = os.path.join(args.output_dir, args.cls_model_save_name)
    if not args.cls_overwrite_output_dir:
        try:
            logging.info("Trying to load classification model if it exists..")
            device = torch.device("cuda")
            model = AutoModelForSequenceClassification.from_pretrained(classification_model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(classification_model_path)
            logging.info(f"Successfully loaded classification model and tokenizer from {classification_model_path}.")
            return {"model": model, "tokenizer": tokenizer}

        except OSError:
            logging.info("Classification model not found. Will need to train new model.")
            pass

    logging.info(f"Loading pretrained model {args.cls_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.cls_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.cls_model).to('cuda')
    logging.info("Model and tokenizer loaded.")

    logging.info("Tokenizing and creating validation and train dataset..")
    train_texts, train_labels = separate_labels(train_data)
    train_dataset = TextDataset(args, tokenizer, train_texts, train_labels)
    val_texts, val_labels = separate_labels(val_data)
    val_dataset = TextDataset(args, tokenizer, val_texts, val_labels)
    logging.info(f"Created train ({len(train_dataset)} examples) and valid ({len(val_dataset)} examples) dataset.")

    training_args = TrainingArguments(
        output_dir=classification_model_path,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=args.cls_per_device_train_batch_size,
        per_device_eval_batch_size=args.cls_per_device_eval_batch_size,
        learning_rate=args.cls_learning_rate,
        num_train_epochs=args.cls_num_train_epochs,
        logging_strategy="steps",
        logging_steps=500,
        logging_dir=os.path.join(args.output_dir, "cls_logs", args.exp_name),
        disable_tqdm=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    logging.info("Training classification model..")
    trainer.train()

    logging.info(f"Saving model to {classification_model_path}")
    trainer.save_model(classification_model_path)

    return {"model": model, "tokenizer": tokenizer}


def run_classification_model(args, classification_model, texts, progress_bar=False):
    """Runs classification model and returns predictions."""

    if len(texts) == 0:
        return []

    model, tokenizer = classification_model["model"], classification_model["tokenizer"]
    model.eval()

    text_dataset = TextDataset(args, tokenizer, texts)
    dataloader = DataLoader(
        text_dataset, batch_size=args.cls_per_device_eval_batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    if progress_bar:
        dataloader = tqdm(dataloader)

    predictions = []
    for batch in dataloader:
        with torch.no_grad():
            move_dict_to_device(batch, "cuda")
            logits = model(**batch).logits
            batch_predictions = logits.argmax(dim=1).tolist()
            predictions.extend(batch_predictions)

    return predictions


def find_leakages(args, classification_model, data, cache=None):                                  
    """Finds the leakages given the classification model and the data."""

    texts = []
    labels = []
    for i in data:
        y, x = i.split(sep="\t")
        texts.append(x)
        labels.append(int(y))
    
    if cache is None:
        predictions = run_classification_model(args, classification_model, texts, progress_bar=True)
    else:
        predictions = [cache["prediction_map"][text] for text in texts]

    leakages = []
    filter_candidates = 0
    for text, label, prediction in zip(texts, labels, predictions):
        if prediction == 1:
            filter_candidates += 1
        elif label == 1 and prediction == 0:
            leakages.append(text)
        else:
            continue

    return leakages, filter_candidates


def get_embedding_model(model):
    """Gets base transformer model from SequenceClassification Model."""

    basenames = ["bert", "roberta", "distilbert"]
    for basename in basenames:
        if hasattr(model, basename):
            return getattr(model, basename)

    raise ValueError("Couldn't find the base model.")


def compute_embeddings(args, texts, embedding_model, convert_to_numpy=False, progress_bar=False):
    """Computes embeddings of texts given an embedding model."""

    model, tokenizer = embedding_model["model"], embedding_model["tokenizer"]
    model.eval()
    text_dataset = TextDataset(args, tokenizer, texts)
    dataloader = DataLoader(
        text_dataset, batch_size=args.cls_per_device_eval_batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    if progress_bar:
        dataloader = tqdm(dataloader)

    embeddings = []

    for batch in dataloader:
        with torch.no_grad():
            move_dict_to_device(batch, "cuda")
            batch_embeddings = model(**batch, output_hidden_states=True).hidden_states[-1][:, 0, :]
            if convert_to_numpy:
                batch_embeddings = batch_embeddings.cpu()
            embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings)

    if convert_to_numpy:
        embeddings = embeddings.numpy()

    return embeddings
