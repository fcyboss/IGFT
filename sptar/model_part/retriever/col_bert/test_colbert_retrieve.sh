#!/bin/bash
helpFunction()
{
   echo ""
   echo "Usage: $0 -g 0,1 -d msmarco -e noaug -p 2000 -c msmarco.psg.l2/checkpoints/colbert-300000.dnn"
   echo -e "\t-g: cudaNum"
   echo -e "\t-d: datasetname, valid datasetnames are: msmarco, hotpotqa, and fiqa"
   echo -e "\t-e: expname, valid expnames are: noaug, inpairs, and sptadr"
   echo -e "\t-p: partitions, the larger the dataset is, the bigger partitions will be. 2000 for msmarco 8M corpus"
   echo -e "\t-c: checkpoint name"
   exit 1 # Exit script after printing help
}

while getopts "g:d:e:p:c:" opt
do
   case "$opt" in
      g ) cudaNum="$OPTARG" ;;
      d ) datasetname="$OPTARG" ;;
      e ) expname="$OPTARG" ;;
      p ) partitions="$OPTARG" ;;
      c ) checkpointname="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$cudaNum" ] || [ -z "$datasetname" ] || [ -z "$expname" ] || [ -z "$partitions" ] || [ -z "$checkpointname" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$cudaNum"
echo "$datasetname"
echo "$expname"
echo "$partitions"
echo "$checkpointname"
# Config path
export cwd="$(pwd)"
export colbert_dir="${cwd}/model_part/retriever/col_bert"
export data_dir="${colbert_dir}/data/datasets/${datasetname}/${expname}/test"
export raw_data_dir="${cwd}/model_part/datasets/raw/beir/${datasetname}"
export model_dir="${colbert_dir}/data/models/${datasetname}/${expname}"

# Mention Any BEIR dataset here (which has been preprocessed)
export dataset="$datasetname"

export CHECKPOINT="${model_dir}/${expname}/train.py/${dataset}.${expname}.l2/checkpoints/colbert-${checkpointname}.dnn"
# Path where preprocessed collection and queries are present
export COLLECTION="${data_dir}/collection.tsv"
export QUERIES="${data_dir}/queries.tsv"
export TRIPLES="${data_dir}/triples.jsonl"

# Path to store the faiss index and run output
export INDEX_ROOT="${model_dir}/test/colbert-${checkpointname}/indices"
export OUTPUT_DIR="${model_dir}/test/colbert-${checkpointname}/output"

# Path to store the rankings file
export RANKING_DIR="${model_dir}/test/colbert-${checkpointname}/rankings"

# Num of partitions in for IVPQ Faiss index (You must decide yourself)
export NUM_PARTITIONS=${partitions}

# Some Index Name to store the faiss index
export INDEX_NAME="${datasetname}"

# 
export LOG_DIR="${model_dir}/test/colbert-${checkpointname}"
# Setting LD_LIBRARY_PATH for CUDA (Incase required!)
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ukp/thakur/cuda/lib64/

################################################################
# 0. BEIR Data Formatting: Format BEIR data useful for ColBERT #
################################################################ 
if [[ $dataset -eq "msmarco" ]]
then
    export test_split="dev"
else
    export test_split="test"
fi

python -m model_part.retriever.col_bert.colbert.data_prep \
    --dataset ${dataset} \
    --split ${test_split} \
    --data_dir $raw_data_dir \
    --collection $COLLECTION \
    --queries $QUERIES \

python -m  model_part.retriever.col_bert.colbert.index \
    --root $OUTPUT_DIR \
    --doc_maxlen 350 \
    --mask-punctuation \
    --bsize 128 \
    --amp \
    --checkpoint $CHECKPOINT \
    --index_root $INDEX_ROOT \
    --index_name $INDEX_NAME \
    --collection $COLLECTION \
    --experiment ${dataset}

CUDA_VISIBLE_DEVICES=0 python -m model_part.retriever.col_bert.colbert.index_faiss \
    --index_root $INDEX_ROOT \
    --index_name $INDEX_NAME \
    --partitions $NUM_PARTITIONS \
    --sample 0.3 \
    --root $OUTPUT_DIR \
    --experiment ${dataset}
    
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=30 python -m model_part.retriever.col_bert.colbert.retrieve_for_al \
    --amp \
    --doc_maxlen 350 \
    --mask-punctuation \
    --bsize 256 \
    --queries $QUERIES \
    --triples $TRIPLES \
    --collection $COLLECTION \
    --nprobe 32 \
    --partitions $NUM_PARTITIONS \
    --faiss_depth 100 \
    --depth 100 \
    --index_root $INDEX_ROOT \
    --index_name $INDEX_NAME \
    --checkpoint $CHECKPOINT \
    --root $OUTPUT_DIR \
    --experiment ${dataset} \
    --ranking_dir $RANKING_DIR
