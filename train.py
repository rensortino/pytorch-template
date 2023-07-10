import torch
from utils.parser import get_parser, setup
from utils.model import read_model_config
from datamodules import MNISTDataModule
from models import Classifier
from trainers import ClsTrainer
from utils import Logger, load_weights, read_model_config
import wandb

def main():
    parser = get_parser() 
    args = parser.parse_args()
    setup(args)

    print(f"Logging on wandb: {args.on_wandb}")        
    wandb.login()
    mode = "online" if args.on_wandb else "disabled"
    wandb.init(name=args.run_name, project=args.wandb_project, mode=mode)

    # Define datamodule
    datamodule_kwargs = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "workers": args.workers,
    }
    
    datamodule = MNISTDataModule(**datamodule_kwargs)

    # Init logger
    logger_kwargs = {
        "output_dir": args.run_dir,
        "run_name": args.run_name,
        "on_wandb": args.on_wandb,
    }

    logger = Logger(**logger_kwargs)

    # Define model
    default_trainer_kwargs = {
        "epochs": args.epochs,
        "save_freq": args.save_freq,
        "log_freq": args.log_freq,
        "debug": args.debug,
        "datamodule": datamodule,
        "logger": logger,
        "lr": args.lr,
    }

    model_kwargs = read_model_config(args.model_config)
    model = Classifier(**model_kwargs)
    model = model.to(args.device)

    if args.resume:
        ckpt = torch.load(args.resume)
        load_weights(model, ckpt)

    # Setup trainer
    trainer = ClsTrainer(model, **default_trainer_kwargs)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
