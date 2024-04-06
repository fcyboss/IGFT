helpFunction()
{
   echo ""
   echo "Usage: $0 -g 0,1 -d msmarco -e noaug -p 2000 -c msmarco.psg.l2/checkpoints/colbert-300000.dnn"
   echo -e "\t-g: cudaNum"
   echo -e "\t-d: datasetname, valid datasetnames are: msmarco, hotpotqa, and fiqa"
   echo -e "\t-e: expname, valid expnames are: noaug, inpairs, and sptadr"
   echo -e "\t-m: max_steps"
   echo -e "\t-s: save checkpoint every s steps"
   echo -e "\t-b: batch size"
   exit 1 # Exit script after printing help
}

while getopts "g:d:e:m:s:b:" opt
do
   case "$opt" in
      g ) cudaNum="$OPTARG" ;;
      d ) datasetname="$OPTARG" ;;
      e ) expname="$OPTARG" ;;
      m ) max_steps="$OPTARG" ;;
      s ) save_checkpoint_steps="$OPTARG" ;;
      b ) batchsize="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if [ -z "$cudaNum" ] || [ -z "$datasetname" ] || [ -z "$expname" ] || [ -z "$max_steps" ] || [ -z "$save_checkpoint_steps" ] || [ -z "$batchsize" ] 
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$cudaNum"
echo "$datasetname"
echo "$expname"
# Config path
export cwd="$(pwd)"
export colbert_dir="${cwd}/model_part/retriever/col_bert"
export data_dir="${colbert_dir}/data/datasets/${datasetname}/${expname}/train" 
export raw_data_dir="${cwd}/model_part/datasets/raw/beir/${datasetname}" 
export model_dir="${colbert_dir}/data/models/${datasetname}/${expname}"

# Mention Any BEIR dataset here (which has been preprocessed)
export dataset="$datasetname"

# Path where preprocessed collection and queries are present
export COLLECTION="${data_dir}/collection.tsv"
export QUERIES="${data_dir}/queries.tsv"
export TRIPLES="${data_dir}/triples.jsonl"
echo $COLLECTION
echo $QUERIES
echo $TRIPLES
echo $model_dir
echo $expname

export INDEX_NAME="${datasetname}"

export TOKENIZERS_PARALLELISM=true




python -m model_part.retriever.col_bert.colbert.sample \
    --amp \
    --maxsteps ${max_steps} \
    --save_checkpoint_steps ${save_checkpoint_steps}\
    --doc_maxlen 350 \
    --mask-punctuation \
    --bsize ${batchsize}\
    --accum 1 \
    --triples $TRIPLES \
    --collection $COLLECTION \
    --queries $QUERIES \
    --root $model_dir \
    --experiment ${expname} \
    --similarity l2\
    --run "${dataset}.${expname}.l2"\
    --checkpoint "colbert-1000.dnn"