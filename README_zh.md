# 参考项目链接

llama工厂

https://github.com/hiyouga/LLaMA-Factory.git

sptar

https://github.com/zhiyuanpeng/SPTAR.git

# 项目流程

首先使用SFT脚本开始训练llama模型，接着使用微调好的llama模型生成pseudo query，之后以此使用BM25、pretrained DPR、lossnet对pseudo query进行过滤，之后将数据喂给ColBERT进行训练(这一步我们使用的是SPTAR的框架)，之后根据ColBERT的召回结果，对pseudo query的难度进行评判，最后将数据通过Reward Model和Proximal Policy Optimization进行强化学习，并重复上述步骤

# SFT数据集准备

数据集下载链接：

https://github.com/beir-cellar/beir

我们使用了FiQA、MSMARCO和NQ数据集，之后将数据集转化成llama工厂可以接受的数据形式，我们将原数据集的corpus.jsonl、queries.jsonl和qrels/train.tsv转化成训练json数据集，形式如下：

```json
{

"instruction": corpus,

"input": "",

"output": query,

"history": []

}
```

之后将数据集文件放入/data文件夹下，将文件名加入到dataset_info.json中，如下

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

其中$name\_sft\_dataset$是需要设定的数据集名称，path_sft_dataset.json是文件夹名称

# SFT llama

将数据集转化成上述的数据集形式后，使用以下脚本进行微调llama模型

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

其中$model\_name\_or\_path$是llama模型的地址，$sft\_checkpoint\_model$是训练出的模型的地址，N是使用的GPU数量

# 生成pseudo query

使用以下脚本可以生成pseudo query

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

其中，$query\_per\_document$代表为每个document生成几个pseudo query，$sample\_num$代表从unlabeled corpus中随机选择多少corpus为其生成对应的pseudo query，generate_path代表生成的pseudo query的json文件的地址，$unlabeled\_corpus\_id$代表未标注的数据集的id的json文件，$corpus\_file$代表该文件的corpus文件

# 预训练DPR过滤器

在SPTAR文件夹下，我们使用以下命令训练出预训练的DPR过滤器

```bash
# datasetname = fiqa, msmarco, nq
python model_part/dpr_eval.py --dataset_name datasetname --version v1 --gpu_id 0 --train_num 50 -exps no_aug
```

$datasetname$代表数据集的名称

# Sparse过滤和Dense过滤

在生成pseudo query后，我们首先需要对生成的pseudo query的质量进行过滤，对此，我们使用BM25和预训练DPR来实现这一点，具体代码如下：

```bash
python src/data_filter.py \
--dataset datasetname \
--sample_doc_num  300 \
--corpus_file corpus_file_path \
--generare_sptar_path generate_data_for_sptar \
--dpr dpr_model \
--pseudo_query_file query_tobe_filtered
```

其中$sample\_doc\_num$代表筛选的粒度，$corpus\_file$代表该数据集的corpus文件,$generate\_data\_for\_sptar$代表生成sptar数据的文件夹，$dpr\_model$代表参与Dense过滤的DPR模型的地址，$query\_tobe\_filtered$代表需要过滤的pseudo query文件

# LossNet过滤

在使用了保证了pseudo query质量的质量过滤后，接下来，我们使用lossnet对pseudo query对colbert的训练必要性进行过滤，lossnet的训练部分代码如下：

```bash
# gen ColBERT data (You need to run this command in py37 env)
# datasetname = fiqa,msmarco,nq
python model_part/retriever/dpr/train/gen_data_for_colbert.py --dataset_name datasetname --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
# train lossnet for active learning
bash model_part/retriever/col_bert/train_colbert_al.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m 500 -s 500 -b 32
```

$dataset\_name$是数据集名称

使用lossnet对数据集打分的代码如下

```bash
# fiqa
# gen ColBERT data (You need to run this command in py37 env)
python zhiymodel_partuan/retriever/dpr/train/gen_data_for_colbert.py --dataset_name datasetname --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
# train
bash model_part/retriever/col_bert/sample_colbert.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m 10000 -s 10000 -b 1

```

# ColBERT训练和测试

在这里，我们使用的是sptar的代码框架进行训练，它们训练代码如下

```bash
# generate data for training and testing
python model_part/retriever/dpr/train/gen_data_for_colbert.py --dataset_name datasetname --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
# train the colbert
bash model_part/retriever/col_bert/train_colbert.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m max_step -s save_step -b batch_size
```

其中，datasetname是你需要训练的数据集的名称，包括fiqa、msmarco和nq；

$max\_step$是训练的step数，$save\_step$是每隔多少step存储一次，$batch\_size$即为训练过程中的批次大小

在测试过程中，我们使用以下代码：

```bash
bash model_part/retriever/col_bert/test_colbert.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -p 200 -c step_model
```

其中$step\_model$代表第几次step保存的模型

# 根据召回结果计算难度

我们基于课程学习的思路，根据训练出的colbert模型对corpus的召回结果计算出难度

```bash
bash model_part/retriever/col_bert/test_colbert_retrieve.sh -g 0 -d datasetname -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -p 200 -c 2000
```

$datasetname$是数据集的名称

# 基于RM和PPO的强化学习

在经过上述的操作，我们可以得到按照难度排序带有可对比属性的数据集，如下：

之后我们将该文件加入到dataset_info.json中，

注意需要加上ranking:True

之后使用如下脚本进行RM和PPO的训练

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

其中，$rm\_1th\_checkpoint\_model$是第一次迭代训练的RM模型，$ppo\_1th\_checkpoint\_model$是第一次迭代训练的PPO模型.