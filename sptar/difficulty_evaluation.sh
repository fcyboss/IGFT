# datasetname = fiqa, msmarco, nq
bash model_part/retriever/col_bert/test_colbert_retrieve.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -p 200 -c 2000