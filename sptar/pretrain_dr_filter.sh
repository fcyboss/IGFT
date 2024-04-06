# datasetname = fiqa, msmarco, nq
python model_part/dpr_eval.py --dataset_name datasetname --version v1 --gpu_id 0 --train_num 50 -exps no_aug
