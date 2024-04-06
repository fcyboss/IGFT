CUDA_VISIBLE_DEVICES=0 python src/query_generate_da_fiqa.py \
    --model_name_or_path llama_model_path \
    --checkpoint_dir model_llama_checkpoint\
    --template default \
    --query_per_document 2 \
    --sample_num 20000 \
    --generate_path pseudo_query.json \
    --unlabeled_corpus_id unlabeled_corpus_id.json \
    --corpus_file dataset_corpus_file

