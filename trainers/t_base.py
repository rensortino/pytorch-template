from tqdm import tqdm


class BaseTrainer:

    def __init__(self, logger, datamodule, lr, epochs=300, best_ckpt_metric=None, start_epoch=0, save_freq=100, log_freq=5, debug=False):
        r'''
        Basic training class to be inherited by all trainers.

        Args:
            logger_kwargs: (utils.logging.Logger) logger instance
            datamodule: (datamodules.base.BaseDataModule) object instance inheriting from datamodules.base.BaseDataModule
            epochs: (int) number of epochs to train for
            best_ckpt_metric: (str) key for the metric to use to select the best checkpoint
            start_epoch: (int) initial epoch number, useful for resuming training
            save_freq: (int) frequency at which to save intermediate checkpoints
            log_freq: (int) frequency at which to log last checkpoints
            debug: (bool) whether to run in debug mode
        '''

        self.logger = logger
        self.train_loader = datamodule.train_loader
        self.test_loader = datamodule.test_loader
        self.lr = lr
        self.num_epochs = epochs
        self.start_epoch = start_epoch
        self.best_ckpt_metric = best_ckpt_metric
        if self.best_ckpt_metric is not None:
            self.best_ckpt = {self.best_ckpt_metric: float("inf")}

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.debug = debug

    def _make_ckpt(self, epoch):
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
        }

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            # Run one epoch of training
            losses_epoch = self.train_one_epoch()

            # Log training metrics
            for key in losses_epoch.keys():
                self.logger.log_metric(f'train/{key}', losses_epoch[key], epoch)

            # Run one epoch of evaluation
            losses_epoch = self.eval_one_epoch()

            # Log evaluation metrics
            for key in losses_epoch.keys():
                self.logger.log_metric(f'eval/{key}', losses_epoch[key], epoch)

            self.visualize_results(epoch)

            # Save checkpoints
            ckpt = self._make_ckpt(epoch)
            if self.best_ckpt_metric is not None:
                ckpt[self.best_ckpt_metric] = losses_epoch[self.best_ckpt_metric]
                self.update_best_ckpt(ckpt)

            if (epoch + 1) % self.log_freq == 0:  # Save last checkpoint
                self.logger.save_ckpt(ckpt, f'last.pt')

            if (epoch + 1) % self.save_freq == 0:  # Save intermediate checkpoint
                self.logger.save_ckpt(ckpt, f'{self.logger.run_name}-{epoch+1}.pt')
                if self.best_ckpt_metric is not None:
                    self.logger.save_ckpt(self.best_ckpt, f'best.pt')

    def train_one_epoch(self):
        epoch_losses = {}
        self.model.train()
        for batch in tqdm(self.train_loader):
            batch_losses = self.training_step(batch)
            if epoch_losses == {}:
                # Initialize epoch_losses keys
                epoch_losses = {k: 0 for k in batch_losses}
            epoch_losses = {k: v + batch_losses[k]
                            for k, v in epoch_losses.items()}
            if self.debug:
                break

        # Aggregating and logging metrics
        epoch_losses = {k: v / len(self.train_loader)
                        for k, v in epoch_losses.items()}
        return epoch_losses

    def training_step(self, batch):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def eval_one_epoch(self):
        raise NotImplementedError

    def eval_step(self, batch):
        raise NotImplementedError
    
    def visualize_results(self, epoch):
        raise NotImplementedError

    def _is_better_than(self, ckpt, best_ckpt):
        if self.best_ckpt_metric is None:
            return False
        return ckpt[self.best_ckpt_metric] < best_ckpt[self.best_ckpt_metric]

    def update_best_ckpt(self, ckpt):
        if self._is_better_than(ckpt, self.best_ckpt):
            self.best_ckpt = ckpt
