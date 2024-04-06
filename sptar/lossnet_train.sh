# gen ColBERT data (You need to run this command in py37 env)
# datasetname = fiqa,msmarco,nq
python model_part/retriever/dpr/train/gen_data_for_colbert.py --dataset_name datasetname --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
# train lossnet for active learning
bash model_part/retriever/col_bert/train_colbert_al.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m 500 -s 500 -b 32
