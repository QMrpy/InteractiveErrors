""" WARNING: Untested Code: GPT2 model not available in Texar Cloud repository"""

import argparse
import logging
import json
import random
import numpy as np
import torch
import torch.nn as nn
import texar.torch as tx
from texar.torch.run import *


class ConditionalGPT2Model(nn.Module):
  
  def __init__(self, args, vocab_size):
    super().__init__()
    self.embedder = tx.modules.WordEmbedder(vocab_size=vocab_size)
    self.encoder = tx.modules.RoBERTaEncoder(pretrained_model_name="roberta-large", cache_dir=args.cls_model)
    self.decoder = tx.modules.GPT2Decoder(pretrained_model_name="gpt2-medium", cache_dir=args.gen_model)

  def _get_decoder_output(self, batch, train=True):
    BOS_TOKEN_ID = 1
    
    enc_states = self.encoder(inputs=self.embedder(batch['source_text_ids']), sequence_length=batch['source_length'])
    if train:  
      return self.decoder(
          inputs=batch['target_text_ids'], sequence_length=batch['target_length'] - 1,
          memory=enc_states, memory_sequence_length=batch['source_length'])
    
    else:      
      start_tokens = torch.full_like(batch['source_text_ids'][:, 0], BOS_TOKEN_ID)
      return self.decoder(
          beam_width=5, start_tokens=start_tokens,
          memory=enc_states, memory_sequence_length=batch['source_length'])

  def forward(self, batch):
    outputs = self._get_decoder_output(batch)
    loss = tx.losses.sequence_sparse_softmax_cross_entropy( 
        labels=batch['target_text_ids'][:, 1:], logits=outputs.logits,
        sequence_length=batch['target_length'] - 1)  

    return {"loss": loss}

  def predict(self, batch):
    sequence, _ = self._get_decoder_output(batch, train=False)

    return {"gen_text_ids": sequence}


def main(args):
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    data_hparams = {
		'train': {
			'source_dataset': {'files': args.train_source_file, 'vocab_file': args.source_vocab_file},
			'target_dataset': {'files': args.train_target_file, 'vocab_file': args.target_vocab_file},
			'batch_size': 8,
			'seed': RANDOM_SEED
		}, 
		'test': {
			'source_dataset': {'files': args.test_source_file, 'vocab_file': args.source_vocab_file},
			'target_dataset': {'files': args.test_target_file, 'vocab_file': args.target_vocab_file},
			'batch_size': 16,
			'seed': RANDOM_SEED
		},
		'valid': {
			'source_dataset': {'files': args.valid_source_file, 'vocab_file': args.source_vocab_file},
			'target_dataset': {'files': args.valid_target_file, 'vocab_file': args.target_vocab_file},
			'batch_size': 16,
			'seed': RANDOM_SEED
		}
	}
    datasets = {split: tx.data.PairedTextData(hparams=data_hparams[split]) for split in ["train", "valid", "test"]}

    model = ConditionalGPT2Model(args, datasets["train"].target_vocab.size)

    executor = Executor(
    model=model, datasets=datasets,
    optimizer={"type": torch.optim.Adam, "kwargs": {"lr": 5e-4}},
    stop_training_on=cond.epoch(20),
    log_every=cond.iteration(1000),
    validate_every=cond.epoch(5),
    train_metric=("loss", metric.RunningAverage(10, pred_name="loss")),
    valid_metric=metric.BLEU(pred_name="gen_text_ids", label_name="target_text_ids"),
    save_every=cond.validation(better=True),
    checkpoint_dir=args.model_save_dir)

    executor.train()
    executor.test(datasets["test"])


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)

	parser = argparse.ArgumentParser()
	parser.add_argument("--train_source_file", type=str, help="Path of train source dataset")
	parser.add_argument("--train_target_file", type=str, help="Path of train target dataset")
	parser.add_argument("--test_source_file", type=str, help="Path of test source dataset")
	parser.add_argument("--test_target_file", type=str, help="Path of test target dataset")
	parser.add_argument("--valid_source_file", type=str, help="Path of validation source dataset")
	parser.add_argument("--valid_target_file", type=str, help="Path of validation target dataset")
	parser.add_argument("--source_vocab_file", type=str, help="Path to classifier or encoder vocabulary file")
	parser.add_argument("--target_vocab_file", type=str, help="Path to generator or decoder vocabulary file")
	parser.add_argument("--cls_model", type=str, default="roberta-large", help="Path or name of classifier model")
	parser.add_argument("--gen_model", type=str, default="gpt2-medium", help="Path or name of generator model")
	parser.add_argument("--model_save_dir", type=str, help="Path to save fine tuned model to")
	args = parser.parse_args()

	main(args)