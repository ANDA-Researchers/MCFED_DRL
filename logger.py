from torch.utils.tensorboard import SummaryWriter
import wandb


class WandbLogger:
    def __init__(self, configs):
        self.logger = wandb.init(
            project="mcfed",
            config=configs,
        )

    def log(self, metric, value, step):
        wandb.log({metric: value}, commit=True)


class TensorboardLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.logger = SummaryWriter(log_dir=save_dir, flush_secs=1)

    def log(self, metric, value, step):
        self.logger.add_scalar(metric, value, step)
