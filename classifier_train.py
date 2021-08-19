"""Takes in training data, and fine tunes a classifier on that"""

import argparse
import os
import random
import logging
import torch
import time
import numpy as np

from utils import load_data, split_data
from classification import train_classification_model


def seed(RANDOM_SEED=42):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)


def main(args):
    RANDOM_SEED = 42
    device = "cuda"

    seed(RANDOM_SEED)

    input_file = os.path.join(args.input_dir, args.input_data)

    logging.info("Loading data and shuffling")
    data = load_data(input_file, shuffle=True)

    logging.info("Splitting data into train, validation and seed datasets")
    train_data, val_data, _ = split_data(data, [0.8, 0.199, 0.001], seed=RANDOM_SEED)
    logging.info(f"Created train ({len(train_data)} examples) and validation ({len(val_data)} examples) dataset.")

    logging.info("Fine tuning classification model")
    _ = train_classification_model(args, train_data, val_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".", help="Directory with input data")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory where to save output")
    parser.add_argument("--input_data", type=str, help="Input file containing sentences in [score, query] format")
    parser.add_argument("--exp_name", type=str, default="roberta-large-fine-tune")
    parser.add_argument("--cls_model", type=str, default="roberta-large")
    parser.add_argument("--cls_overwrite_output_dir", action="store_true")
    parser.add_argument("--cls_per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--cls_per_device_eval_batch_size", type=int, default=2048)
    parser.add_argument("--cls_learning_rate", type=float, default=1e-5)
    parser.add_argument("--cls_num_train_epochs", type=int, default=3)
    parser.add_argument("--cls_max_seq_length", type=int, default=40)
    parser.add_argument("--cls_model_save_name", type=str, default="roberta-large")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()

    logging_level = logging.INFO if not args.quiet else logging.ERROR
    logging.basicConfig(level=logging_level)

    main(args)

