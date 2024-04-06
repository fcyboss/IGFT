# fiqa
# gen ColBERT data (You need to run this command in py37 env)
python model_part/retriever/dpr/train/gen_data_for_colbert.py --dataset_name datasetname --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
# train
bash model_part/retriever/col_bert/sample_colbert.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m 10000 -s 10000 -b 1


