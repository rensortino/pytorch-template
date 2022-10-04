from evaluators.e_classifier import ClsEvaluator
from utils.arg_parser import get_args_parser, setup
from pathlib import Path
from logzero import logger as lz_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datamodules.base import BaseDataModule
from models.m_classifier import Classifier
from trainers.t_classifier import Trainer
from utils.logging import Logger
import json
from utils.misc import count_parameters, fix_seed

import torch
from attrdict import AttrDict

import wandb

def main(args):

    fix_seed(args.seed)

    # Data Loading
    dm = BaseDataModule(data_dir='data')
    train_loader = dm.train_dataloader(args.batch_size, args.device)
    if not args.one_batch:
        val_loader = dm.val_dataloader(args.batch_size, args.device)
        test_loader = dm.test_dataloader(args.batch_size, args.device)
    else:
        lz_logger.warning('Training on a single batch')


    logger = Logger(args.output_dir, args.run_name)

    trainer_kwargs = {
        'one_batch': args.one_batch,
    }
    trainer = Trainer(args.resume, scheduler, **trainer_kwargs)
    evaluator = ClsEvaluator()
    if args.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(trainer.opt, patience=50)
    elif args.lr_scheduler == 'none':
        scheduler = None

    print('*'*70)
    lz_logger.info(f'Logging general information')
    lz_logger.info(f'Debugging mode: {args.debug}')
    lz_logger.info(f'Model total parameters {(count_parameters(trainer.model) / 1e6):.2f}M')
    lz_logger.info(f'Model trainable parameters {(count_parameters(trainer.model, trainable=True) / 1e6):.2f}M')
    print('*'*70)
            
    try:
        if args.inference:
            assert args.resume, 'No weights selected for inference'
            epoch = args.start_epoch
            loss = evaluator.test_one_epoch(test_loader)
            lz_logger.info(f'Inference loss: {loss}')
            return

        # Training loop
        for epoch in range(trainer.start_epoch, args.epochs):
            lz_logger.info(f'Training [{epoch} / {args.epochs}]')
            train_loss = trainer.train_one_epoch(train_loader)
            if (epoch + 1) % 10 == 0:
                logger.save_ckpt(trainer.model.state_dict())
            logger.log_metric('train/loss', train_loss, epoch)
            lz_logger.info(f'Epoch {epoch}: {train_loss}')
            
            if not (args.debug or args.one_batch):
                ckpt = {
                    'epoch': epoch,
                    'gen' : trainer.model.gen.state_dict(),
                    'disc' : trainer.model.disc.state_dict(),
                }
                logger.save_ckpt(ckpt)
            if args.one_batch:
                continue

            lz_logger.info(f'Validation [{epoch} / {args.epochs}]')
            val_loss = evaluator.validate_one_epoch(trainer.model, val_loader)
            logger.log_metric(f'val/loss', val_loss, epoch)

            if args.lr_scheduler != 'none':
                trainer.scheduler.step(val_loss)

            logger.log_lr(trainer.opt, epoch)

            if (epoch + 1) % args.log_every_n_epochs == 0:
                best_ckpt = trainer.get_best_ckpt(val_loss, epoch)
                logger.save_ckpt(best_ckpt, 'best.pt')

            if (epoch + 1) % args.test_every_n_epochs == 0:
                evaluator.test_one_epoch(test_loader)

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