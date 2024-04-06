import os
import time
import faiss
import random
import torch
import itertools
import shutil
import json
import csv
from tqdm import tqdm

from colbert.utils.runs import Run
from multiprocessing import Pool
from colbert.modeling.inference import ModelInference
from colbert.evaluation.ranking_logger import RankingLogger

from colbert.utils.utils import print_message, batch
from colbert.ranking.rankers import Ranker



def read_json_dict(args):
    json_file = args.triples
    qcid_dict = {}
    qid_list = []
    cid_list = []
    with open(json_file,'r') as f:
        lines = f.readline() 
        while lines:
            lines = json.loads(lines)
            qid ,cid = lines[0], lines[1]
            qcid_dict[qid] = cid
            lines = f.readline()

   
    return qcid_dict



def read_tsv_corpus(args):
    tsv = args.collection 
    cid_count_dict = {}
    count = 0
    with open(tsv, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for row in tsvreader:
            cid_count_dict[row[0]]  = count
            count += 1
    return cid_count_dict


def retrieve_for_al(args):
    inference = ModelInference(args.colbert, amp=args.amp)
    ranker = Ranker(args, inference, faiss_depth=args.faiss_depth)
    rank_qid_dict = {}
    ranking_logger = RankingLogger(Run.path, qrels=None)
    milliseconds = 0


    qcid_dict = read_json_dict(args)
    cid_count_dict = read_tsv_corpus(args)

    hit_num = 0
    all_num = 0

    with ranking_logger.context('ranking.tsv', also_save_annotations=False) as rlogger:
        queries = args.queries
        qids_in_order = list(queries.keys())
        result = {}
        # qids_count_dict = {}
        # qcount = 0
        # for qid in qids_in_order:
        #     qids_count_dict[qid] = qcount
        #     qcount += 1

        for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
            qbatch_text = [queries[qid] for qid in qbatch]
            qbatch_dict = {queries[qid]:qid for qid in qbatch}
            
            rankings = []
            for query_idx, q in enumerate(qbatch_text):
                torch.cuda.synchronize('cuda:0')
                s = time.time()

                Q = ranker.encode([q])
               
                label_qid = qbatch_dict[q]
                label_pid = qcid_dict[label_qid]
                count_label_id = cid_count_dict[label_pid]
                pids,scores = ranker.rank(Q)

                all_num += 1
                


                if count_label_id in pids:
                    
                    rank_index = pids.index(count_label_id)
                    score = scores[rank_index]
                    yuan_corpus_id = label_pid
                    count_id = pids[rank_index]
                    rank_index += 1 #
                    hit_num += 1
                    print(f"hit! {hit_num}/{all_num}")
                else:
                    rank_qid_dict[label_qid] = 0
                    rank_index = -1 
                    score = 0
                    yuan_corpus_id = label_pid
                    print(f"sorry!{all_num}")
                if yuan_corpus_id in result.keys():
                    result[yuan_corpus_id].append({'rank_index':rank_index,'score':score,'yuan_corpus_id':yuan_corpus_id})
                else:
                    result[yuan_corpus_id] = [{'rank_index':rank_index,'score':score,'yuan_corpus_id':yuan_corpus_id}]





               

                torch.cuda.synchronize()
                milliseconds += (time.time() - s) * 1000.0

        with  open('diff.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)



    print(hit_num)

    print('\n\n')
    print(ranking_logger.filename)
    if args.ranking_dir:
        os.makedirs(args.ranking_dir, exist_ok=True)
        shutil.copyfile(ranking_logger.filename, os.path.join(args.ranking_dir, "ranking.tsv"))
        print("#> Copied to {}".format(os.path.join(args.ranking_dir, "ranking.tsv")))
    print("#> Done.")
    print('\n\n')
