import os
import random
import torch
import copy
import sys
from os.path import join
cwd = os.getcwd()
colbert_dir = join(cwd, "model_part", "retriever", "col_bert")
if colbert_dir not in sys.path:
    sys.path.append(colbert_dir)
import colbert.utils.distributed as distributed

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run
from colbert.sampling.sampling import sample


def main():
    parser = Arguments(description='Sample the most valuable data')

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_training_input()

    args = parser.parse()

    assert args.bsize % args.accumsteps == 0, ((args.bsize, args.accumsteps),
                                               "The batch size must be divisible by the number of gradient accumulation steps.")
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512

    args.lazy = args.collection is not None

    with Run.context(consider_failed_if_interrupted=False):
        sample(args)


if __name__ == "__main__":
    main()
