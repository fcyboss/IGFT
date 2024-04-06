import json
from tqdm import tqdm
import random
import os
import argparse
from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import random
import logging
import pathlib, os
import random
import argparse
from os.path import join
import sys
from tqdm import tqdm
import logging
import sys
import torch
from typing import Dict, List
logger = logging.getLogger(__name__)
import torch
import numpy as np
import csv
from rank_bm25 import BM25Okapi


# DensePassageRetrieval:
class DensePassageRetrieval():
    def __init__(self,dpr_path): # model_save_path means the path of DPR
        self.model_save_path = dpr_path 
        self.model = DRES(models.SentenceBERT(model_save_path), batch_size=256, corpus_chunk_size=10000)
        self.retriever = EvaluateRetrieval(self.model, k_values=[1,3,5,10], score_function="cos_sim")

    def retrieve_score(self,queries,corpus):
        if len(queries) < 2:
            queries['aaa'] = 'aaa'
        results = self.retriever.retrieve(corpus, queries)
        return results

# BM25:
class RankBM():
    def __init__(self,corpus) -> None:
        self.corpus = corpus
        self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25_corpus = BM25Okapi(self.tokenized_corpus)
    def get_score(self,query):
        return self.bm25_corpus.get_scores(query.split(" "))
    def get_top_doc(self, query, n = 1):
        return self.bm25_corpus.get_top_n(query.split(" "),self.corpus,n)
    
    

# run

def adhere_id(qd_file, corpus_file):
    print("begin search id")
    corpus_data = {}
    with open(corpus_file, 'r') as corpus_file:
        for line in tqdm(corpus_file):
            data = json.loads(line)
            corpus_data[data['text']] = data['_id']
    with open(qd_file, 'r') as fiqa_file:
        fiqa_data = json.load(fiqa_file)
    for item in tqdm(fiqa_data):
        document = item['document']
        item['_id'] = "-1"
        if document in corpus_data.keys():
            item['_id'] = corpus_data[document]
    with open(qd_file, 'w') as fiqa_file:
        json.dump(fiqa_data, fiqa_file, indent=2)
    print("done!")



def get_random_doc(dict1,key1, sample_doc_num = 10):
    result = [dict1[key1]]
    random_keys = random.sample([key for key in dict1 if key!= key1], sample_doc_num)
    result.extend([dict1[i] for i in random_keys  ])
    return result



def select_with_bm(file_with_id,sample_doc_num):
    print(f"begin select query with BM25, the number of sample: {sample_doc_num}")
    good_question_count = 0
    count = 0
    result = []
    file = file_with_id
    with open(file,'r') as f:
        data = json.load(f)
    id_corpus = {}
    query_id = {}
    for i in tqdm(data):
        id = i['_id']
        corpus = i['document']
        if type(i['good_question']) == str:
            query = i['good_question']
        else:
            query = i['good_question'][0]
        if id not in id_corpus.keys():
            id_corpus[id] = corpus
        query_id[query] = id
    for query in tqdm(query_id.keys()):
        ids = query_id[query]
        random_doc = get_random_doc(id_corpus, ids,sample_doc_num)
        bm1 = RankBM(random_doc)
        score = bm1.get_score(query)
        score = list(score)
        max_index = score.index(max(score))
        if max_index == 0: # hit!
            good_question_count += 1
            result.append({"_id":ids,'good_question':query,'document':random_doc[0],'score':score[0]})
        count += 1
    with open(file_with_id, 'w') as fiqa_file:
        json.dump(result, fiqa_file, indent=2)
    print("done!")





def select_with_dpr(file_with_id,sample_doc_num,dataset,dpr_path):
    print("begin select query with DPR")
    good_question_count = 0
    split_alpha = 25
    count = 0
    json_result = []
    dpr = DensePassageRetrieval(dpr_path)
    file = file_with_id
    with open(file,'r') as f:
        data = json.load(f)
    id_corpus = {}
    query_id = {}
    for i in tqdm(data):
        id = i['_id']
        corpus = i['document']
        if type(i['good_question']) == list:

            query = i['good_question'][0]
        else:
            query = i['good_question']
        if id not in id_corpus.keys():
            id_corpus[id] = corpus
        query_id[query] = id

    print("number of query:,", len(query_id))   
    chunk_length = len(query_id)//split_alpha
    print('chunk_length: ',chunk_length)
    chunks_queries = [list(query_id.keys())[i:i+chunk_length] for i in range(0,len(query_id),chunk_length)]
    chunks_queries[-1].extend(list(query_id.keys())[split_alpha*chunk_length:])

    score_result = []
    for itr_count,chunk_query in enumerate(chunks_queries):
        print(f"{itr_count}/{len(chunks_queries)}, chunk_length:{len(chunk_query)}")
        random.shuffle(chunk_query)
        query_for_dpr = {}
        corpus_for_dpr = {}
        count += len(chunk_query)
        for query in tqdm(chunk_query):
            ids = query_id[query]
            query_for_dpr[ids] = query
            c_id = f'corpus_{ids}'
            c = id_corpus[ids]
            corpus_for_dpr[c_id] = {'text':c,'title':''}

        result_chunk = dpr.retrieve_score(query_for_dpr, corpus_for_dpr)

        for query in tqdm(chunk_query):
            ids = query_id[query]
            score = result_chunk[ids]


            score = process_score_few_shot(score,ids)
            if score != 0: 
                good_question_count += 1
                c = id_corpus[ids]
                score_result.append({"_id":ids,'good_question':query,'document':c,'score':score})
        print(f"ratio:{good_question_count/count}")
    with open(file_with_id, 'w') as fiqa_file:
        json.dump(score_result, fiqa_file, indent=2)
    print("done!")

def process_score_few_shot(dict1,ids):
    assert len(dict1) != 0
    values = list(dict1.values())
    max_value = max(values)
    truth_value = dict1.get(f'corpus_{ids}',0)
    return truth_value if truth_value == max_value else 0
  

def generate_query_tsv(file_with_id, query_file,score_tsv):
    max_query_id = 0
    max_corpus_id = 0
    print("Begin generate")
    with open(file_with_id, 'r', encoding='utf-8') as data_file:
        data = json.load(data_file)

    new_queries = []
    new_corpus = []
    max_query_id = 2000001
    data = [i for i in data if i['score']!=0]

    for item in data:
        max_query_id += 1
        max_corpus_id += 1
        query_item = {
            '_id': str(max_query_id),
            'text': item['good_question'], 
            'metadata': {}
        }
        new_queries.append(query_item)
    triplets = []

    for i in range(len(new_queries)):
        query_id = new_queries[i]['_id']
        corpus_id = data[i]['_id']
        score = 1
        triplet = (query_id, corpus_id, score)
        triplets.append(triplet)

    select_num = min(len(triplets),100000)
    random_triplets = random.sample(triplets, select_num) # 

    with open(score_tsv, 'w', encoding='utf-8') as dev_file:
        for triplet in random_triplets:
            query_id, corpus_id, score = triplet
            dev_file.write(f'{query_id}\t{corpus_id}\t{score}\n')

    with open(query_file, 'w', encoding='utf-8') as queries_file:
        for query_item in new_queries:
            queries_file.write(json.dumps(query_item, ensure_ascii=False) + '\n')





def main(args):

    dataset = args.dataset
    sample_doc_num = args.sample_doc_num

    corpus_file = args.corpus_file
    generare_sptar_path = args.generare_sptar_path
    dpr_path = args.dpr_path
    pseudo_query_file = args.pseudo_query_file

    query_file = f'{generare_sptar_path}/weak_queries_50_llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70.jsonl'
    score_tsv = f'{generare_sptar_path}/weak_train_50_llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70.tsv'
    if not os.path.exists(f'{generare_sptar_path}'):
        os.mkdir(f'{generare_sptar_path}')
    adhere_id(pseudo_query_file, corpus_file)
    select_with_bm(pseudo_query_file,sample_doc_num)
    select_with_dpr(pseudo_query_file,sample_doc_num,dataset,dpr_path)
    generate_query_tsv(pseudo_query_file, query_file, score_tsv) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,default='fiqa')
    parser.add_argument("--sample_doc_num", type=int,default=100)

    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--generare_sptar_path", type=str)
    parser.add_argument("--dpr_path", type=str)
    parser,add_argument("--pseudo_query_file",type=str)



    args = parser.parse_args()



    main(args)
