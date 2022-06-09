from arg_parser import get_args_parser, setup
from pathlib import Path
from logzero import logger as lz_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datamodules.mnist import MNISTDataModule
from models.m_classifier import Classifier
from trainers.t_classifier import Trainer
from utils.logging import Logger
import json
from utils.misc import count_parameters, eliminate_randomness

import torch
from attrdict import AttrDict

import wandb

def main(args):

    eliminate_randomness(args.seed)

    # Data Loading
    dm = MNISTDataModule(data_dir='data')
    train_loader = dm.train_dataloader(args.batch_size)
    if not args.one_batch:
        val_loader = dm.val_dataloader(args.batch_size)
        test_loader = dm.test_dataloader(args.batch_size)
    else:
        lz_logger.warning('Training on a single batch')

    model = Classifier(args)
    opt = torch.optim.Adam(model.parameters(), args.lr)
    if args.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(opt, patience=50)
    elif args.lr_scheduler == 'none':
        scheduler = None

    logger = Logger(args)

    trainer_kwargs = {
        'one_batch': args.one_batch,
        'log_every_n_epochs': args.log_every_n_epochs,
        'log_weights': args.log_weights,
        'log_gradients': args.log_gradients,
        'batch_size': args.batch_size,
        'debug': args.debug
    }
    trainer = Trainer(model, opt, logger, args.resume, scheduler, **trainer_kwargs)

    print('*'*70)
    lz_logger.info(f'Logging general information')
    lz_logger.info(f'Debugging mode: {args.debug}')
    lz_logger.info(f'Model total parameters {(count_parameters(model) / 1e6):.2f}M')
    print('*'*70)
            
    try:
        if args.inference:
            assert args.resume, 'No weights selected for inference'
            epoch = args.start_epoch
            # TODO Replace with evaluator
            loss = trainer.test_one_epoch(test_loader)
            lz_logger.info(f'Inference loss: {loss}')
            return

        # Training loop
        for epoch in range(args.start_epoch, args.epochs):
            trainer.train_one_epoch(train_loader)
            if (epoch + 1) % 10 == 0:
                logger.save_ckpt(model.state_dict())

            if args.one_batch:
                trainer.increment_epoch()
                continue

            trainer.validate_one_epoch(val_loader)

            logger.log_lr(opt, epoch)

            if (epoch + 1) % args.test_every_n_epochs == 0:
                trainer.test_one_epoch(test_loader)
            trainer.increment_epoch()

    except KeyboardInterrupt as ki:
        print('Interrupted by user')
        wandb.finish()
        return

    
if __name__ == '__main__':

    args = get_args_parser().parse_args()
    
    run_mode = 'disabled' if args.debug else 'online'  
    wandb.init(
            config=args,
            name=args.run_name,
            mode=run_mode
        )
    
    setup(args)
    main(args)