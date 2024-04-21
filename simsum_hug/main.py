'''
Main Program: 
> CUDA_VISIBLE_DEVICES=x python main.py --dataset=<your dataset>
'''
# -- fix path --
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#torch.multiprocessing.set_start_method('forkserver', force=True)
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
# -- end fix path --

from preprocessor import WIKI_DOC, D_WIKI, EXP_DIR, MILDSUM
import time
import json

from argparse import ArgumentParser
import pytorch_lightning as pl


"""
    Select the model before execution by uncomenting the required model and commenting the rest
"""
# from T5_2 import SumSim as model, train
#from Bart2 import SumSim as model, train
from Bart_baseline_finetuned import BartBaseLineFineTuned as model, train
# from T5_baseline_finetuned import T5BaseLineFineTuned as model, train


def parse_arguments():
    p = ArgumentParser()
                  
    p.add_argument('--seed', type=int, default=42, help='randomization seed')
    p.add_argument('--dataset', default='MILDSUM', type=str, help="select dataset")

    p = model.add_model_specific_args(p)
    p = pl.Trainer.add_argparse_args(p)
    args,_ = p.parse_known_args()
    return args

# class MetricsCallback(pl.Callback):
#   def __init__(self):
#     super().__init__()
#     self.metrics = []
  
#   def on_validation_end(self, trainer, pl_module):
#       self.metrics.append(trainer.callback_metrics)

# Create experiment directory
def get_experiment_dir(create_dir=False):
    dir_name = f'{int(time.time() * 1000000)}'
    path = EXP_DIR / f'exp_{dir_name}'
    if create_dir == True: path.mkdir(parents=True, exist_ok=True)
    return path

def log_params(filepath, kwargs):
    filepath = Path(filepath)
    kwargs_str = dict()
    for key in kwargs:
        kwargs_str[key] = str(kwargs[key])
    json.dump(kwargs_str, filepath.open('w'), indent=4)



def run_training(args):

    args.output_dir = get_experiment_dir(create_dir=True)
    print(f"Experment Dir: {args.output_dir}")
    # logging the args
    log_params(args.output_dir / "params.json", vars(args))

    if args.dataset == 'WIKI_DOC':
        args.dataset = WIKI_DOC
    elif args.dataset == 'MILDSUM':
        args.dataset = MILDSUM
    else:
        args.dataset = D_WIKI
    print("Dataset: ",args.dataset)
    train(args)


if __name__ == "__main__":
    args = parse_arguments()
    run_training(args)