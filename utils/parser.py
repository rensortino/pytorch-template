import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # Dataset options
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data-dir', default='data', type=str)
    parser.add_argument('--workers', type=int, default=0)
    # Training options
    parser.add_argument('--model-config')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--resume')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:0')
    # Logging options
    parser.add_argument('--run-dir', type=str, default='runs/')
    parser.add_argument('--run-name', type=str, default='placeholder')
    parser.add_argument('--on-wandb', type=bool, default=True)
    parser.add_argument('--wandb-entity', type=str,
                        default="rensortino")
    parser.add_argument('--wandb-project', type=str,
                        default="pytorch-template")
    parser.add_argument('--log-freq', type=int, default=10)
    parser.add_argument('--save-freq', type=int, default=150)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--best-ckpt-metric', type=str, help="Key for metric to use to save best checkpoint")
    return parser

def setup(args):

    if args.debug:
        args.workers = 0
        args.batch_size = 2
        args.epochs = 2
        args.log_freq = 1
        args.save_freq = 1
        args.on_wandb = False
        args.run_name = "debug"