# gen ColBERT data (datasetname = fiqa, msmarco, nq)
python model_part/retriever/dpr/train/gen_data_for_colbert.py --dataset_name datasetname --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
# # # train ( -m: max steps -s:save step -b: batch size)
bash model_part/retriever/col_bert/train_colbert.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m 3000 -s 100 -b 32

