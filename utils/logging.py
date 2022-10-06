from datetime import datetime
from pathlib import Path
from torchsummary.torchsummary import summary
import sys
import torch
import wandb
from logzero import logger as lz_logger
import logzero

class Logger:

    def __init__(self, output_dir, run_name, wandb_run=None):
        
        self.out_dir = Path(output_dir)
        self.run_name = Path(run_name)
        self._set_dirs()
        self.wandb_run = wandb_run

        logzero.logfile(self.out_dir / Path('output.log'))
        wandb.login()      

        self._log_command()

    def _set_dirs(self):
        current_date = datetime.now()
        formatted_date = f'{current_date.year}-{current_date.month}-{current_date.day}-{current_date.hour}h{current_date.minute}'
        test_name = f'{formatted_date}'
        self.out_dir = self.out_dir / self.run_name / Path(test_name)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _log_command(self):
        with open(self.out_dir / Path('command.txt'), 'w') as out:
            cmd_args = ' '.join(sys.argv)
            out.write(cmd_args)
            out.write('\n')

    def log_image(self, pil_img, phase, title, step, on_wandb=True):
        pil_img.save(self.out_dir / Path(f'{title}.png'))
        if on_wandb and self.wandb_run:
            wandb.log({f'{phase}/{title}': wandb.Image(pil_img), 'epoch':step})

    def log_metric(self, title, metric, step, on_wandb=True):
        lz_logger.info(f'Epoch [{step}] - {title}: {metric}')
        if on_wandb and self.wandb_run:
            wandb.log({title: metric, 'epoch': step})

    def log_summary(self, model, input_size=(4,), tgt_size=(4,260), batch_size=32):
        # Give no batch size in tgt_size
        summary(model, self.out_dir / Path('summary.txt'), input_size, tgt_size, batch_size)

    def log_lr(self, opt, epoch, title='train/lr', on_wandb=True):
        lr = get_lr(opt)
        if on_wandb and self.wandb_run:
            wandb.log({title: lr, "epoch": epoch})

    def warning(self, text):
        lz_logger.warning(text)

    def info(self, text):
        lz_logger.info(text)

    def save_ckpt(self, ckpt, path='checkpoint.pt', on_wandb=True):
        path = Path(path)
        lz_logger.info(f'Saving {path} in {self.out_dir}')
        torch.save(ckpt, self.out_dir / path)
        if on_wandb and self.wandb_run:
            model = wandb.Artifact('model', type='model')
            model.add_file(self.out_dir / path)
            self.wandb_run.log_artifact(model)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    logger = Logger('logs_test', 'Test logging')
    logger.log_metric('Test metric', 0.5, 1, on_wandb=False)
    logger.info('This is an info message')
    logger.warning('This is a warning')
    ckpt = torch.randn(64, 3, 256, 256)
    logger.save_ckpt(ckpt, 'test.pt', on_wandb=False)