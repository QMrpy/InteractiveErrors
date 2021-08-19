# python bart_masking.py --input_file ../data/src_covid_us/Covid.txt \
#     --output_file h1_mask_bart.json \
#     --model_path ~/Projects/IEOutputs/h1_bart/

# python bart.py --queries_file ../data/Hate_Data/mixed_queries \
#     --leakages_file ../data/src_covid_us/Covid.txt \
#     --output_file output-file-name.json \
#     --output_dir ~/Projects/IEOutputs \
#     --model_path ~/Projects/IEOutputs/h1_bart_both/ \
#     --train_samples 100000 \
#     --learning_rate 2e-5 \
#     --per_device_train_batch_size 16 \
#     --k 10

# python bart_masking.py --input_file ../data/src_covid_us/US_election_seed_queries.txt \
#     --output_file usel_h1_mask_bart_pos.json \
#     --model_path ~/Projects/IEOutputs/h1_bart

# python bart_masking.py --input_file ../data/src_covid_us/US_election_seed_queries.txt \
#     --output_file usel_h1_mask_bart_all.json \
#     --model_path ~/Projects/IEOutputs/h1_bart_both

python data_handling_and_evaluation/compute_metrics_h1.py --input_file usel_h1_mask_bart_pos.json.txt --input_file_tsv --lm_path gpt2-medium --score_up 0.989 --score_lo 0.7

python data_handling_and_evaluation/compute_metrics_h1.py --input_file usel_h1_mask_bart_all.json.txt --input_file_tsv --lm_path gpt2-medium --score_up 0.989 --score_lo 0.8

# python compute_metrics_h1.py --input_file ~/Projects/IEOutputs/json_outputs/h1_bart_all/h1_mask_bart.json.txt --input_file_tsv --lm_path gpt2-medium --score_up 0.8 --score_lo 0.7

# python compute_metrics_h1.py --input_file ~/Projects/IEOutputs/json_outputs/h1_bart_all/h1_mask_bart.json.txt --input_file_tsv --lm_path gpt2-medium --score_up 0.7 --score_lo 0.6

# python compute_metrics_h1.py --input_file ~/Projects/IEOutputs/json_outputs/h1_bart_all/h1_mask_bart.json.txt --input_file_tsv --lm_path gpt2-medium --score_up 0.6 --score_lo 0.5

