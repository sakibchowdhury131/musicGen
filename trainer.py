from model import LitAutoEncoder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
import yaml


with open('config.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

autoencoder = LitAutoEncoder(train_dir= config['train_dir'], val_dir=config['train_dir'], lr = config['learning_rate'], batch_size = config['batch_size'])

logger = TensorBoardLogger("tensorboard", name="decoder_with_attention")

checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints", 
                                      save_top_k=2, 
                                      monitor="validation_loss",
                                      mode="min",)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback],
    devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
    limit_train_batches = config['limit_train_batches'],
    max_epochs=config['max_epochs'],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    logger = logger,
    #resume_from_checkpoint = "/content/checkpoints/epoch=0-step=1103.ckpt"
)


trainer.fit(model=autoencoder)
