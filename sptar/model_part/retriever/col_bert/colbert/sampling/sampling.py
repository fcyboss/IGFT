import os
import random
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager
import json
from colbert.sampling.lazy_batcher import LazyBatcher
from colbert.sampling.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.sampling.utils import print_progress, manage_checkpoints


from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples


class LossNet(nn.Module):
    def __init__(self,dim = 128,n_heads=4):
        super(LossNet, self).__init__()
        self.cross_atten = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads).to(DEVICE)
        self.linear1 = nn.Linear(32*128, 32*32).to(DEVICE)
        self.linear2 = nn.Linear(32*32, 32*4).to(DEVICE)
        self.linear3 = nn.Linear(32*4, 32*1).to(DEVICE)
        self.linear4 = nn.Linear(32*1, 1).to(DEVICE)

        self.line_contrast = nn.Linear(2,1).to(DEVICE)


    def forward(self,Q,D):
        Q = Q.transpose(0, 1)
        D = D.transpose(0, 1)
        attn_output, _ = self.cross_atten(Q, D, D)
        attn_output = attn_output.view(attn_output.shape[1], -1)
        attn_output = self.linear1(attn_output)
        attn_output = self.linear2(attn_output)
        attn_output = self.linear3(attn_output)
        attn_output = self.linear4(attn_output)
        attn_output = torch.mean(attn_output)
        return attn_output



class ScoreBatcher(): 
    def __init__(self, args):

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)

        self.bsize = args.bsize
    def _split_into_batches(self,ids, mask, bsize):
        batches = []
        for offset in range(0, ids.size(0), bsize):
            batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))
        return batches
    def encode(self,query, pos,neg):
        query, doc, neg_doc = [query], [pos], [neg]
        N = len(query)
        Q_ids, Q_mask = self.query_tokenizer.tensorize(query)
        D_ids, D_mask = self.doc_tokenizer.tensorize(doc + neg_doc)
        D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

        maxlens = D_mask.sum(-1).max(0).values


        indices = maxlens.sort().indices
        Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
        D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

        (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

        query_batches = self._split_into_batches(Q_ids, Q_mask, self.bsize)
        positive_batches = self._split_into_batches(positive_ids, positive_mask, self.bsize)
        negative_batches = self._split_into_batches(negative_ids, negative_mask, self.bsize)

        batches = []
        for (q_ids, q_mask), (p_ids, p_mask), (n_ids, n_mask) in zip(query_batches, positive_batches, negative_batches):
            Q = (torch.cat((q_ids, q_ids)), torch.cat((q_mask, q_mask)))
            D = (torch.cat((p_ids, n_ids)), torch.cat((p_mask, n_mask)))
            batches.append((Q, D))

        return batches[0]






def sample(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)


    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    if args.lazy:
        reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    scoreBatcher = ScoreBatcher(args)

    colbert = ColBERT.from_pretrained('bert-base-uncased', #'bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)



    if args.checkpoint is not None:
        print('loading colbert ........')
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)

    lossnet = torch.load("lossnet.pth") # lossnet path



    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']

        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])
    Q_set = set()
    loss_score_result = []


    count = 0



    query_text_list = []
    pos_text_list = []
    neg_text_list = []

    print("begin load dataset")
    count = 0
    for BatchSteps in tqdm(reader):
        query_text, pos_text, neg_text = BatchSteps
        query_text_list.append(query_text)
        pos_text_list.append(pos_text)
        neg_text_list.append(neg_text)
        if count>=len(reader):
            break
        count += 1

    assert len(query_text_list) == len(pos_text_list) == len(neg_text_list)

    for i in tqdm(range(len(query_text_list))):
        query, pos, neg = query_text_list[i][0], pos_text_list[i][0], neg_text_list[i][0]
        if query + pos in Q_set:
            continue
        Q_set.add(query+pos)
        pos_batch = scoreBatcher.encode(query, pos, neg)
        Q,D = pos_batch
        with torch.no_grad():
            _,Q,D = colbert(Q,D)
            predict_loss = lossnet.sample(Q,D).item()

        loss_score_result.append({"good_question":query,'document':pos,'loss_score':predict_loss})

    with open('loss_score.json','w') as f:
        json.dump(loss_score_result, f, indent=2)
