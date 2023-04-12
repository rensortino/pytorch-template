import argparse
from attrdict import AttrDict
import yaml

def get_args_parser():
    parser = argparse.ArgumentParser()

    # * Training Parameters
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--test_every_n_epochs', type=int, default=15,
                        help='Frequency to run tests')

    # * Model parameters
    parser.add_argument('--config', default='classifier.yaml')
    # * Optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--opt', default="adam", choices=['adam', 'sgd'], type=str)
    parser.add_argument('--lr_scheduler', default="none", choices=['none', 'plateau'], type=str)
    
    # * Logger parameters
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--run_name', default='placeholder')
    parser.add_argument('--on-wandb', type=bool, default=True)
    parser.add_argument('--log-freq', type=int, default=5, 
                                help="Frequency for image logging")
    parser.add_argument('--save-freq', type=int, default=100, help="Frequency for saving checkpoints")
    
    # * Other parameters
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)

    return parser


def setup(args):

    if args.debug:
        args.run_name = 'debug'
        args.log_freq = 1
        args.test_every_n_epochs = 1
        args.on_wandb = False
        args.batch_size = 2
        args.output_dir = 'output/debug'