from llmtuner import ChatModel, create_app
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation.utils import GenerationConfig
import random
from tqdm import tqdm
import json
import os
import torch
import argparse
from typing import Any, Dict, Optional, Tuple

import json
import csv
import sys
from llmtuner.tuner.core import get_train_args, get_infer_args, load_model_and_tokenizer

def get_filter(filter_file):
    id_list = []
    with open(filter_file,'r', newline= '') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter ='\t')
        for row in reader:
            id_list.append(row['_id'])
    return id_list


def read_json(filename):
    text_dict = {}
    with open(filename, 'r', encoding='utf-8') as jsonfile:
        for line in jsonfile:
            data = json.loads(line)
            data_id = data['_id']
            data_text = data['text']
            text_dict[data_id] = data_text
    return text_dict



def get_corpus(corpus_file):
    c_dict = read_json(corpus_file)
    return c_dict



def  generate_query(model,query_per_document,filter_id,generate_path,corpus_file):

    c_dict = get_corpus(corpus_file)
    corpus_max_length = 20000


    generated_qd = []
    for count in tqdm(filter_id):
        corpus = c_dict[count]
        print(len(generated_qd))
        for _ in range(query_per_document):
            try:
                response = model.chat(corpus)
                print(response)
            except torch.cuda.OutOfMemoryError:
                print(f"raise the OutOfMemoryError, the corpus is {len(corpus)}")
            temp = {"good_question":response,"document":corpus}
            generated_qd.append(temp)  
            
         
    with open(generate_path, 'a', encoding='utf-8') as f:
        json.dump(generated_qd, f, ensure_ascii=False, indent=2)
    print("generate done")   
    return generated_qd






def main(args: Optional[Dict[str, Any]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--query_per_document', type=int, required=True)
    parser.add_argument('--sample_num', type=int, required=True)
    parser.add_argument('--generate_path', type=str, required=True)

    parser.add_argument("--unlabeled_corpus_id", type=str, required = True)
    parser.add_argument("--corpus_file", type=str, required = True)

    args = parser.parse_args()
    query_per_document = args.query_per_document
    sample_num = args.sample_num
    generate_path = args.generate_path
    unlabeled_corpus_id = args.unlabeled_corpus_id
    corpus_file = args.corpus_file
    filter_id = get_filter(unlabeled_corpus_id)
    delattr(args, 'query_per_document')
    delattr(args, 'sample_num')
    delattr(args, 'generate_path')
    delattr(args, 'unlabeled_corpus_id')
    delattr(args, 'corpus_file')

    args = vars(args)
    model = ChatModel(args)
    generated_qd = generate_query(model, query_per_document,filter_id,generate_path,corpus_file)


if __name__ == "__main__":
    main()
   


