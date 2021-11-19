# python run_full_data.py <ROOT:str> <EXP:str> <RUN:int>

from data_diet.fair_train import train
import sys
from types import SimpleNamespace
from jax.config import config
config.update("jax_debug_nans", True)

# setup
ROOT = sys.argv[1]
EXP = sys.argv[2]
RUN = int(sys.argv[3])
META_MODEL_SEED, META_TRAIN_SEED, SEED_INCR = 117, 117117, 711
EP_STEPS = 390
DATA_DIR = ROOT + '/data'
EXPS_DIR = ROOT + '/exps'

# arguments
args = SimpleNamespace()
# data
args.data_dir = DATA_DIR
args.dataset = EXP
# subsets
args.subset = None
args.subset_size = None
args.scores_path = None
args.subset_offset = None
args.random_subset_seed = None
# model
args.model = 'mlp'
args.model_seed = META_MODEL_SEED + RUN * SEED_INCR
args.load_dir = None
args.ckpt = 0
# optimizer
args.lr = 0.001
args.beta = 0.9
args.weight_decay = 0.0005
args.nesterov = True
args.lr_vitaly = False
args.decay_factor = 0.2
args.decay_steps = [60*EP_STEPS, 120*EP_STEPS, 160*EP_STEPS]
# training
args.num_steps = 200*EP_STEPS
args.train_seed = META_TRAIN_SEED + RUN * SEED_INCR
args.train_batch_size = 128
args.test_batch_size = 20
args.augment = False
args.track_forgetting = False
# checkpoints
args.save_dir = EXPS_DIR + f'/{EXP}/fair_{RUN}'
args.log_steps = EP_STEPS
args.early_step = 0
args.early_save_steps = None
args.save_steps =  EP_STEPS

# experiment
train(args)
