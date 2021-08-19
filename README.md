# Code For Explore-Search in Interactive Errors

1. `bart_rl_hyperparameter_search.yaml, t5_rl_hyperparameter_search.yaml, gpt2_rl_hyperparameter_search.yaml`: YAML files to run H2 for the three different generators on Amulet. **Warning:** Amulet doesn't save outputs and networkx fails with *PowerIterationFailedConvergence* for a large horizon length, i.e., greater than 150 steps.

2. `bart.py`: Fine tunes *facebook/bart-large* model on given data, and/or generates candidate leakages from given leakages, by randomly masking spans in the sentence.

3. `classification.py, classification_with_leakage_filtering.py`: Almost same files, except `find_leakages` function is different; in the later one it returns the filtered candidates too (sentences with label 0). Has classifier utilities to fine-tune and run classification model.

4. `classifier_train.py`: Takes in training data and fine tunes Transformers classifier on the data

5. `encoder_decoder.py`: Fine tunes and trains an encoder decoder model for H3, using model cross attentions. Uses BERT as encoder and GPT2 as decoder. The model has not been trained yet as losses kept oscillating.

6. `gpt2.py`: Generates candidate leakages from *gpt2-medum* trained on queries or tweets.

7. `graph_rl.py`: Driver code for RL search on graph, using any algorithm from *OpenAI stable_baselines3==1.0*. Can use any transformer environment `gpt2env.py, bartenv.py, t5env.py`.

8. `mixer_decoder_modified.py`: Contains the mixer and classifier (encoder) - generator (decoder). Uses `past_key_values` as the decoder input after encoder context or `hidden_states` has been transformed through the mixer. **Warning:** Incomplete Code.

9. `sampling.py`: Does a sampling of `hidden_states` of the classifier using Gaussian Processes or assuming a pointwise Gaussian for every `hidden_state`. Gaussian Processes use **pymc3** and the code runs quite slow.

10. `t5.py`:  Fine tunes *t5-base* model on given data, and/or generates candidate leakages from given leakages, by masking Noun Phrases in the sentence.

11. `utils.py`: Utilities used for classification. Required by `classification.py` everywhere.

12. `bart_masking.py`: Takes in sentences, and masks important POS Tags (Verb, Noun, Adverb, Adjective, Proper Noun). Generates all possible combinations of masked templates, and fills the masks using fine tuned BART. The generated sentences are compared with the semantic similarity to the original sentence, and the masked templates are sorted in decreasing order of average semantic similarity to the original sentence. Higher ranked templates are better in preserving the intent of the query, and in generating queries close to the original query.


# Setting up the environment

1. Create a conda virtual environment in GCR machine

```
source /anaconda/bin/activate
conda activate py37_default
```

2. Install `requirements.txt` and other required packages

```
pip install -r requirements.txt
pip install matplotlib sklearn networkx
```

# Running unconditional generation (H1)

1. First T5, BART, GPT2 need to be fine tuned. The files `t5.py, bart.py` have a flag `--no_train` which is to be used from next runs. GPT2 needs a different fine tuning script. To fine tune BART and then generate from it, using default parameters use,

```
python bart.py --queries_file /path/to/queries/file \
    --leakages_file /path/to/leakages/file \
    --output_file output-file-name.json \
    --output_dir /path/to/output/dir \
    --model_path /path/where/to/save/trained/model \
    --train_samples 50000 \
    --k 10
```

An output json file will be generated, which is to be annotated and evaluated.

Use `--no_train` to prevent training when the models have already been once fine tuned. `t5.py` is identical in behavior, and it also takes in `--spacy_model`, which can be set to a different language other than English, for example Japanese `ja_core_web_lg`. The model to be used then is mT5 (Multilingual T5).


# Running RL search on graph (H2)

For running RL search using T5, for example, use,

```
python graph_rl.py --input_file /path/to/input/file \
    --output_file output-file-name.json \
    --cls_model /path/to/finetuned/classifier \
    --gen_model /path/to/finetuned/generator-t5 \
    --batch_size 64 \
    --top_k 200 \
    --steps 5000 \
    --horizon 100 \
    --sim_threshold 0.6 \
    --gen_model_name t5-base
```

By default, output files will be generated in the current directory `.`, with an `episode_id` prefixed, such as `1_outputs.json`. Keep `--horizon` less than 200, else, `networkx` fails with PowerIterationFailedConvergence.


# Running Encoder Decoder Model (H3)

To fine tune, save and generate from bert-gpt2 encoder-decoder, use,

```
python encoder_decoder.py --queries_file /path/to/queries/file \
    --leakage_file /path/to/leakages/file \
    --output_file /path/to/outputs.json \
    --model_dir /path/to/save/finetuned/model \
    --train_dir /path/to/log/dir 
```

# Evaluating generated outputs

To create a file containing all generated sentences to be annotated,

```
python data_handling_and_evaluation/process_json_outputs.py --input_dir /path/to/json/outputs \
    --output_file /path/to/file/to/be/annotated \
    --samples 500
```

This generates a file with `500` generated sentences to be annotated. If `--input_dir` is the name of a file, the script processes that file only, else, it processes all json files in the given directory.

After annotations are done, to find the offensive and leakage percentages, use,

```
python data_handling_and_evaluation/find_leakages.py --input_file /path/to/input/file \
    --output_leakage_file /path/to/true/leakages \
    --cls_model /path/to/finetuned/classifier
```

This prints to terminal the offensive, true leakages and filtered leakages percentages.

To compute automated metrics for H1 or H2, such as BLEU, Self-BLEU, Perplexity, Semantic Similarities, use `compute_metrics_h1.py` or `compute_metrics_h2.py`, and load finetuned GPT2 from its path. They take in the original json output files. For `compute_metrics_h2.py`,

```
python compute_metrics_h2.py --input_data /path/to/all/episodes/outputs \
    --lm_path /path/to/finetuned/gpt2
```

Details of data handling and evaluation are there in `data_handling_and_evaluation` *README.md*.
