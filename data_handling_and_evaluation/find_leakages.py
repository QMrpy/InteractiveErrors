import logging
import os
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from classification import find_leakages


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with open(args.input_file, "r") as f:
        data = f.read().splitlines()

    class1_count = 0
    for i in data:
        y, _ = i.split(sep='\t')
        if y == '1':
            class1_count += 1
    logging.info(f"Percentage of offensive is {100 * class1_count / len(data)}%")

    cls_model = AutoModelForSequenceClassification.from_pretrained(args.cls_model).to("cuda")
    cls_tokenizer = AutoTokenizer.from_pretrained(args.cls_model)

    leakages, filter_candidates = find_leakages(args, {"model": cls_model, "tokenizer": cls_tokenizer}, data)
    
    with open(args.output_leakage_file, "w") as f:
        for i in leakages:
            i = i.strip()
            i = i.replace("\n", "")
            f.write(i + "\n")


    logging.info(f"Percentage of true leakages for classifier generated = {(len(leakages) / len(data)) * 100} %")
    logging.info(f"Percentage of filtered leakages for classifier generated = {(len(leakages) / (len(data) - filter_candidates)) * 100} %")


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Annotated input file with generated sentences")
    parser.add_argument("--output_leakage_file", type=str, help="True leakages generated output file")
    parser.add_argument("--cls_model", type=str, default="roberta-large", help="Path or name of classifier model")
    parser.add_argument("--cls_per_device_eval_batch_size", type=int, default=128, help="Keep same as batch size")
    parser.add_argument("--cls_max_seq_length", type=int, default=128, help="Truncate all sequences to this length")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args)