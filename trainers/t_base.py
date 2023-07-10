class BaseTrainer:

    def __init__(self, logger, datamodule, resume=None, device='cuda',
                 epochs=300, save_freq=100, log_freq=5, debug=False):
        r'''
        Basic training logic to be inherited by all trainers.

        Args:
            logger_kwargs: (utils.logging.Logger) logger instance
            datamodule: (datamodules.base.BaseDataModule) object instance inheriting from datamodules.base.BaseDataModule
            resume: (str) checkpoint path to resume from
            device: (str) device to run training on
            epochs: (int) number of epochs to train for
            save_freq: (int) frequency at which to save intermediate checkpoints
            log_freq: (int) frequency at which to log last checkpoints
            debug: (bool) whether to run in debug mode
        '''

        self.logger = logger
        self.datamodule = datamodule
        self.init_checkpoint(resume)
        self.device = device
        self.num_epochs = epochs
        self.start_epoch = 0
        self.best_ckpt = {"loss": float("inf")}

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.debug = debug

    def init_checkpoint(self, resume):
        if resume:
            self.load_checkpoint(resume)

    def load_checkpoint(self, ckpt_path):
        pass

    def train(self, train_loader):
        pass

    def train_one_epoch(self, train_loader):
        pass

    def training_step(self, batch):
        pass

    def update_best_ckpt(self, ckpt):
        if ckpt["loss"] < self.best_ckpt['loss']:
            self.best_ckpt = ckpt