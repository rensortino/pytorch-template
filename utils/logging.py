from datetime import datetime
from pathlib import Path
# from torchsummary.torchsummary import summary
import sys
import torch
import wandb
from logzero import logger as lz_logger
import logzero


# TODO Remove wandb_run and artifacts 
class Logger:

    def __init__(self, output_dir, run_name, on_wandb):
        
        self.out_dir = Path(output_dir)
        self.run_name = Path(run_name)
        self.on_wandb = on_wandb
        self._set_dirs()

        logzero.logfile(self.out_dir / Path('output.log'))
        if on_wandb:
            wandb.login()
            wandb.init(name=run_name)
        
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

    def log_image(self, pil_img, phase, title, step):
        pil_img.save(self.out_dir / Path(f'{title}.png'))
        if self.on_wandb:
            wandb.log({f'{phase}/{title}': wandb.Image(pil_img), 'epoch':step})

    def log_metric(self, title, metric, step):
        lz_logger.info(f'Epoch [{step}] - {title}: {metric}')
        if self.on_wandb:
            wandb.log({title: metric, 'epoch': step})

    def log_lr(self, opt, epoch, title='train/lr'):
        lr = get_lr(opt)
        if self.on_wandb:
            wandb.log({title: lr, "epoch": epoch})

    def warning(self, text):
        lz_logger.warning(text)

    def info(self, text):
        lz_logger.info(text)

    def save_ckpt(self, ckpt, path='checkpoint.pt'):
        path = Path(path)
        lz_logger.info(f'Saving {path} in {self.out_dir}')
        torch.save(ckpt, self.out_dir / path)
        if self.on_wandb:
            wandb.save(str(self.out_dir / path))


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