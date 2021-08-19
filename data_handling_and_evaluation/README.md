# Scripts to handle data and post-process results

1. `compute_metrics_h1.py`: Computes BLEU, Self-BLEU, Perplexity and Semantic Similarities for outputs of H1. GPT2 is used for perplexity calculation.

2. `compute_metrics_h2.py`: Computes BLEU, Self-BLEU, Perplexity and Semantic Similarities for outputs of H2. GPT2 is used for perplexity calculation.

3. `find_leakages.py`: After annotations are done, finds percentage of offensive sentences, leakages and filtered leakages using fine-tuned RoBERTa-large as the proxy classifier.

4. `lm_scores.py`: Finds Language model (BART, T5, GPT2) scores for a given golden annotated dataset.

5. `ping_bing_hate_model.py`: Gets hate scores from the Highway hosted network for a given set of queries.

6. `preprocess_tweets.py`: Preprocesses and cleans tweets using *tweet-preprocessor* and *regex* libraries.

7. `process_json_outputs.py`: Aggregates all generated outputs of all episodes for the RL search (H2) and produces a text file.

8. `process_rl_amulet_logs.py`: Makes a json file out of Amulet logs, in case Amulet doesn't save outputs due to file paths being messed up. Requires a specific format of data to work.