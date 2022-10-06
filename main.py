from utils.arg_parser import get_args_parser, setup
from utils.logging import Logger
from utils.misc import count_parameters, fix_seed, instantiate_from_config
import wandb
from omegaconf import OmegaConf
from pathlib import Path
def main(args):

    fix_seed(args.seed)
    config = OmegaConf.load((Path('config') / args.config).with_suffix('.yaml'))

    run = wandb.init(
        mode="disabled" if args.debug else "online",
        name=args.run_name,
        config=args,
    )

    # Data Loading
    dm = instantiate_from_config(config.data)
    train_loader = dm.train_dataloader(args.batch_size, args.device)
    if not args.one_batch:
        val_loader = dm.val_dataloader(args.batch_size, args.device)
        test_loader = dm.test_dataloader(args.batch_size, args.device)
    else:
        logger.warning('Training on a single batch')

    logger = Logger(args.output_dir, args.run_name, run)

    trainer_config = config.trainer
    trainer_config['one_batch'] = args.one_batch
    trainer = instantiate_from_config(trainer_config)
    evaluator = instantiate_from_config(config.evaluator)

    print('*'*70)
    logger.info(f'Logging general information')
    logger.info(f'Debugging mode: {args.debug}')
    logger.info(f'Model total parameters {(count_parameters(trainer.model) / 1e6):.2f}M')
    logger.info(f'Model trainable parameters {(count_parameters(trainer.model, trainable=True) / 1e6):.2f}M')
    print('*'*70)
            
    try:
        if args.inference:
            assert args.resume, 'No weights selected for inference'
            epoch = args.start_epoch
            loss = evaluator.test_one_epoch(test_loader)
            logger.info(f'Inference loss: {loss}')
            return

        # Training loop
        for epoch in range(trainer.start_epoch, args.epochs):
            logger.info(f'Training [{epoch} / {args.epochs}]')
            train_loss = trainer.train_one_epoch(train_loader)
            if (epoch + 1) % 10 == 0:
                logger.save_ckpt(trainer.model.state_dict())
            logger.log_metric('train/loss', train_loss, epoch)
            logger.info(f'Epoch {epoch}: {train_loss}')
            
            if args.log_weights:
                logger.weight_histogram_adder(trainer.model.named_parameters(), epoch)

            if args.log_gradients:
                logger.gradient_histogram_adder(trainer.model.named_parameters(), epoch)

            if not (args.debug or args.one_batch):
                ckpt = {
                    'epoch': epoch,
                    'gen' : trainer.model.gen.state_dict(),
                    'disc' : trainer.model.disc.state_dict(),
                }
                logger.save_ckpt(ckpt)
            if args.one_batch:
                continue

            logger.info(f'Validation [{epoch} / {args.epochs}]')
            val_loss = evaluator.validate_one_epoch(trainer.model, val_loader)
            logger.log_metric(f'val/loss', val_loss, epoch)

            if args.lr_scheduler != 'none':
                trainer.scheduler.step(val_loss)
            logger.log_lr(trainer.opt, epoch)

            if (epoch + 1) % args.log_every_n_epochs == 0:
                trainer.update_best_ckpt(val_loss, epoch)
                logger.save_ckpt(trainer.best_ckpt, 'best.pt')

            if (epoch + 1) % args.test_every_n_epochs == 0:
                evaluator.test_one_epoch(trainer.model, test_loader)

    except KeyboardInterrupt:
        print('Interrupted by user')
        wandb.finish()
        return

    
if __name__ == '__main__':

    args = get_args_parser().parse_args()
    setup(args)
    main(args)