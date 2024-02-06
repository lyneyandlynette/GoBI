import argparse
import wandb 
from src.collect_data import collect_data
from pdb import set_trace as st

def argparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of RAPID")
    parser.add_argument('--env', type=str, default='MiniGrid-MultiRoom-N7-S4-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_data', type=int, default=100000)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--env_seed', default=1, type=int,
                        help='The seed used to generate the environment if we are using a \
                        singleton (i.e. not procedurally generated) environment.')
    return parser

def main():
    parser = argparser()
    args = parser.parse_args()
    config_dict = vars(args)
    collect_data(args)

if __name__ == '__main__':
    main()

# python collect.py --env MiniGrid-MultiRoom-N7-S8-v0