import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()

    # * Training Parameters
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--one_batch', action='store_true')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--inference', action='store_true', help='Run inference on pretrained model')
    parser.add_argument('--test_every_n_epochs', type=int, default=15,
                        help='Frequency to run tests')

    # * Optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--opt', default="adam", choices=['adam', 'sgd'], type=str)
    parser.add_argument('--lr_scheduler', default="none", choices=['none', 'plateau'], type=str)

    # * Dataset parameters        
    parser.add_argument('--data_dir', default="data/",
                        help="Directory where data is stored")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--persistent_workers', action='store_true')
    
    # * Logger parameters
    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--run_name', default='',
                        help='descriptive name of run')
    parser.add_argument('--log_every_n_epochs', type=int, default=5,
                        help='Frequency to log gradients and images')
    
    # * Other parameters
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    return parser


def setup(args):

    if args.debug:
        args.run_name = 'debug'
        args.log_every_n_epochs = 1
        args.test_every_n_epochs = 1

    if not args.run_name:
        args.run_name = input('Name your run: ')