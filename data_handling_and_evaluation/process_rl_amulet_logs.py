""" 
Input file format should be:
Selected sentence : injury report on barry simmons
Generated sentence 0 : injury rate for gay men
Selected sentence : injury rates for women vs men
Selected sentence : estonian displaced persons
Generated sentence 0 : estonian displaced africans
Generated sentence 1 : estonian displaced people
"""

""" Useful when amulet doesn't save outputs. Some shell preprocessing is needed though """

import argparse
import json

def main(args):

    with open(args.input_file) as f:
        data = f.read().splitlines()

    gen_sentences = []
    for i in range(len(data)):
        if data[i].startswith("Selected sentence :"):
            j = 0
            while data[i + j + 1].startswith("Generated sentence "):
                x = data[i]
                gen_sentences.append({"selected_sentence": x.split(sep=":")[1].strip(), "generated_sentence": data[i + j + 1].split(":")[1].strip()})
                j += 1
                if i + j + 1 >= len(data):
                    break

    with open(args.output_file, "w") as f:
        data = {"output_sentences": gen_sentences}
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Full path of partailly processed logs")
    parser.add_argument("--output_file", type=str, help="Full path of processed json file")
    args = parser.parse_args()

    main(args)
