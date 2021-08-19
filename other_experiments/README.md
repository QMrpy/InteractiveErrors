# Other Experiments done

1. `gpt2_ranked_graph_non_rl.py`: Constructs the sentence similarity graph, and finds out centrality scores (degree or eigenvector). Ranked by centrality scores, sentences are chosen to generate from by the GPT2 generator. It is a different form of H2.

2. `greedy_baselines_ucb.py`: H2 using UCB algorithm. The code is **incomplete** should not be used.

3. `mlm_whole_word_mask.py`: Uses a masked languge model such as RoBERTa to mask whole words instead of word pieces, and generates new sentences by filling the masks. Masks are chosen randomly, according to the BERT paper.

4. `spanbert.py`: Fine tunes *SpanBERT/spanbert-base-cased* model on given data, and/or generates candidate leakages from given leakages, by randomly masking spans in the sentence. The model currently doesn't give legible results, so, the code needs to be fixed.

5. `substitution_mlm_bart.py, substitution_mlm_t5.py`: Ranks words by word attention scores or word gradients (for attentions it is at word level, for gradients it is at subword level), and masks them accordingly in sorted decreeasing order. The generator fills in the masks.

6. `texar_enc_dec.py`: Uses *Texar* library to make an encoder-decoder RoBERTa-GPT2 model. Does not work currently as Texar's hub of models doesn't have GPT2

7. `ucb.py`: Multi-armed bandits Upper Confidence Bound, an alternative to RL search. The *Thompson Sampling* part is incomplete.


# Running Word Substitution with Subword Aggeregation and Attention Score guided masking

To run `substitution_mlm_bart.py` for e.g., use,

```
python substitution_mlm_bart.py --input_file /path/to/leakages \
--output_file /path/to/accepted/outputs \
--gen_output_file /path/to/all/generated/outputs \
--mask_model /path/to/finetuned/bart \
--cls_model /path/to/finetuned/classifier 
```

It uses word attentions by default, to mask and fill masks from there.