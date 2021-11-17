# python get_fair_score.py <ROOT:str> <EXP:str> <RUN:int> <STEP:int> <BATCH_SIZE:int> <TYPE:str>

from data_diet.data import load_data, load_fairness_dataset
from data_diet.scores import compute_fair_scores, compute_scores
from data_diet.utils import get_fn_params_state, load_args
import sys
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root', type=str, default=os.getcwd(),
                    help='dir os root')
parser.add_argument('--exp', type=str, default='CelebA',
                    help='dataset')
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--step', type=int, default=78000)
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('-t', '--type', type=str, default='covariance')

args = parser.parse_args()

run_dir = os.path.join(args.root, f'/exps/{args.exp}/fair_{args.run}')
arg = load_args(run_dir)
arg.load_dir = run_dir
arg.ckpt = args.step

_, X, Y, Z, _, _, _, args = load_fairness_dataset(arg)
fn, params, state = get_fn_params_state(arg)
scores = compute_fair_scores(fn, params, state, X, Y, Z, args.batch_size, args.type)

path_name = 'covariance_scores'

save_dir = os.path.join(run_dir, path_name)
save_path = os.path.join(run_dir, path_name, f'ckpt_{STEP}.npy')
if not os.path.exists(save_dir): os.makedirs(save_dir)
np.save(save_path, scores)