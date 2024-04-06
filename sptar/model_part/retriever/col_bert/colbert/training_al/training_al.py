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

from colbert.training_al.lazy_batcher import LazyBatcher
from colbert.training_al.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training_al.utils import print_progress, manage_checkpoints


import json




import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE
import torch.nn.init as init




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








def train_al(args):
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

    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)



    if args.checkpoint is not None:
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

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()
    
    lossnet = LossNet()
    optimizer_lossnet = AdamW(filter(lambda p: p.requires_grad, lossnet.parameters()), lr=args.lr, eps=1e-8)
    optimizer_lossnet.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()

    criterion_lossnet = nn.MSELoss()

    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0

    if args.resume:
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']
        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    for batch_idx, BatchSteps in zip(tqdm(range(start_batch_idx, args.maxsteps)), reader):
        this_batch_loss = 0.0
        for queries, passages in BatchSteps:
            with amp.context():
                scores,Q,D = colbert(queries, passages)
                predict_loss = lossnet(Q,D)
                loss = criterion(scores, labels[:scores.size(0)])
                loss = loss / args.accumsteps
            
                loss_loss = criterion_lossnet(loss,predict_loss) 

                loss_loss = torch.clamp(torch.mean(loss_loss),min = 0)
                loss = loss + loss_loss 

                
                

            if args.rank < 1:
                print_progress(scores)

            amp.backward(loss)
            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(colbert, optimizer)
        amp.step(lossnet, optimizer_lossnet)


        if (batch_idx + 1) % 100 == 0:
            torch.save(lossnet,f'lossnet_{(batch_idx + 1)//100}.pth')

        if args.rank < 1:
            avg_loss = train_loss / (batch_idx+1)
            num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            log_to_mlflow = (batch_idx % 20 == 0)
            Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

            manage_checkpoints(args, colbert, optimizer, batch_idx+1)


           