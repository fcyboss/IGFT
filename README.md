# Links of reference project

llama factory

https://github.com/hiyouga/LLaMA-Factory.git

sptar

https://github.com/zhiyuanpeng/SPTAR.git

# Project process

The llama model was trained using SFT scripts. Then the fine-tuned llama model was used to generate pseudo queries. The pseudo query was then filtered using BM25, pretrained DPR, and lossnet. The data was then fed to ColBERT for training (we used the SPTAR framework for this step), and the pseudo query was then rated for difficulty based on ColBERT's retrieval results. Finally, the data were reinforced by Reward Model and Proximal Policy Optimization, and the above steps were repeated

# SFT data set preparation

Dataset download link:

https://github.com/beir-cellar/beir

We used FiQA, MSMARCO, and NQ datasets, and then converted the datasets into a form acceptable to the llama factory. We converted the corpus.jsonl, queries.jsonl, and qrels/train.tsv from the original dataset into a training json dataset in the following form:

```json
{

"instruction": corpus,

"input": "",

"output": query,

"history": []

}
```

Then put the dataset file into the /data folder and add the file name to the dataset_info.json file as follows

```json
"name_sft_dataset":{

"file_name": "path_sft_dataset.json",

"columns": {

"prompt": "instruction",

"query": "input",

"response": "output",

"history": "history"

},

"ranking": true

}
```

Where $name\_sft\_dataset$is the name of the dataset to be set, and $path\_sft\_dataset.json$ is the json file name

# SFT llama

After converting the data set into the above data set form, use the following script to fine-tune the llama model

```bash
deepspeed --num_gpus=N src/train_bash.py \
--stage sft \
--model_name_or_path path_model \
--do_train \
--dataset name_sft_dataset \
--template default \
--finetuning_type lora \
--lora_target q_proj,v_proj \
--output_dir sft_checkpoint_model \
--overwrite_cache \
--per_device_train_batch_size 24 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 3000 \
--learning_rate 5e-5 \
--num_train_epochs 3.0 \
--plot_loss \
--fp16 \
--deepspeed ds_config.json
```

Where $model\_name\_or\_path$is the address of the llama model, $sft\_checkpoint\_model$is the address of the trained model and N is the number of Gpus used

# Pseudo query generation

The following script can be used to generate pseudo queries

```bash
CUDA_VISIBLE_DEVICES=0 python src/query_generate_da_fiqa.py \
    --model_name_or_path llama_model_path \
    --checkpoint_dir model_llama_checkpoint\
    --template default \
    --query_per_document 2 \
    --sample_num 20000 \
    --generate_path pseudo_query.json \
    --unlabeled_corpus_id unlabeled_corpus_id.json \
    --corpus_file dataset_corpus_file
```

Where, $query\_per\_document$represents how many pseudo queries are generated for each document, and $sample\_num$represents how many corpus are randomly selected from the unlabeled corpus to generate pseudo queries for each document. generate_path represents the address of the json file for the generated pseudo query, $unlabeled\_corpus\_id$represents the json file with the id of the unlabeled dataset, and $corpus\_file$represents the corpus file for that file

# Pretrain DPR

In the SPTAR folder, we use the following command to train the pre-trained DPR filter

```bash
# datasetname = fiqa, msmarco, nq
python model_part/dpr_eval.py --dataset_name datasetname --version v1 --gpu_id 0 --train_num 50 -exps no_aug
```

$datasetname means the name of dataset

# Sparse filter and Dense filter

After the pseudo query is generated, we first need to filter the quality of the pseudo query generated, and we do this using BM25 and pre-trained DPR, as follows:

```bash
python src/data_filter.py \
--dataset datasetname \
--sample_doc_num  300 \
--corpus_file corpus_file_path \
--generare_sptar_path generate_data_for_sptar \
--dpr dpr_model \
--pseudo_query_file query_tobe_filtered
```

Where $sample\_doc\_num$represents the granularity of the filter,$corpus\_file$represents the corpus file of the data set, and $generate\_data\_for\_sptar$represents the folder where sptar data is generated. $dpr\_model$represents the address of the DPR model participating in Dense filtering, and $query\_tobe\_filtered$represents the pseudo query files tobe filtered

# LossNet Filter

After using a quality filter to ensure the quality of the pseudo query, we then use lossnet to filter the pseudo query's training necessity for colbert, the training part of lossnet is as follows:

```bash
# gen ColBERT data (You need to run this command in py37 env)
# datasetname = fiqa,msmarco,nq
python model_part/retriever/dpr/train/gen_data_for_colbert.py --dataset_name datasetname --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
# train lossnet for active learning
bash model_part/retriever/col_bert/train_colbert_al.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m 500 -s 500 -b 32
```

$dataset\_name$ is the name of dataset

The code to score a dataset using lossnet is as follows

```bash
# fiqa
# gen ColBERT data (You need to run this command in py37 env)
python model_part/retriever/dpr/train/gen_data_for_colbert.py --dataset_name datasetname --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
# train
bash model_part/retriever/col_bert/sample_colbert.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m 10000 -s 10000 -b 1
```

# ColBERT Train and Test

Here, we use sptar's code framework for training, and they train the code as follows

```bash
# generate data for training and testing
python model_part/retriever/dpr/train/gen_data_for_colbert.py --dataset_name datasetname --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
# train the colbert
bash model_part/retriever/col_bert/train_colbert.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m max_step -s save_step -b batch_size
```

datasetname is the name of the dataset you need to train, including fiqa, msmarco, and nq.

$max\_step$is the number of steps trained, $save\_step$is how many steps are stored every other time, and $batch\_size$is the batch size in the training process

During the test, we used the following code:

```bash
bash model_part/retriever/col_bert/test_colbert.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -p 200 -c step_model, 
```

where $step\_model$ represents the model saved in several steps.

# Calculate the difficulty based on the retrieval result

Inspired by course-based learning, we compute the difficulty level using the recall result of a corpus analyzed by the trained Colbert model.

```bash
bash model_part/retriever/col_bert/test_colbert_retrieve.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -p 200 -c 2000
```

$datasetname$ is the name of the dataset.

# RM and PPO based reinforcement learning

After the above operations, we can get a dataset with comparable attributes sorted by difficulty, as follows:

We then add the file to dataset_info.json,

Note the need to add ranking:True

Then use the following script to train RM and PPO

```bash
# RM

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
--stage rm \
--model_name_or_path path_model\
--do_train \
--dataset datasetname \
--template default \
--finetuning_type lora \
--lora_target q_proj,v_proj \
--resume_lora_training False \
--checkpoint_dir sft_checkpoint_model \
--output_dir rm_1th_checkpoint_model \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 1000 \
--learning_rate 1e-8 \
--num_train_epochs 1.0 \
--plot_loss \
--fp16

# PPO

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
--stage ppo \
--model_name_or_path path_model \
--do_train \
--dataset datasetname \
--template default \
--finetuning_type lora \
--lora_target q_proj,v_proj \
--resume_lora_training False \
--checkpoint_dir sft_checkpoint_model \
--reward_model rm_1th_checkpoint_model \
--output_dir ppo_1th_checkpoint_model \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 1000 \
--learning_rate 1e-7 \
--num_train_epochs 1.0 \
--plot_loss \
--bf16
```

Where, $rm\_1th\_checkpoint\_model$is the RM model of the first iteration training, and $ppo\_1th\_checkpoint\_model$is the PPO model of the first iteration training.