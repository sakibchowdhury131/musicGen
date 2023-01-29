import torch
import torch.nn as nn
import os
import pytorch_lightning as pl
from buildSpectrogram import getLogMelSpec
import matplotlib.pyplot as plt
import librosa
from torch.utils.data import Dataset, DataLoader
from torch import optim

class musicDataloader(pl.LightningDataModule):
  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.filelist = os.listdir(self.root_dir)
    #self.x = np.zeros(shape = (len(self.filelist), 128, 698))
    
    #for i, fle in tqdm(enumerate(self.filelist)):
    #  self.x[i] = getLogMelSpec(audio_path = f"{self.root_dir}/{fle}")

    #self.x = torch.from_numpy(self.x)
    #self.n_samples = self.x.shape[0]


  def __len__(self):
    return len(self.filelist)
  
  def __getitem__(self, index):
    if torch.is_tensor(index):
      index = index.tolist()
    

    data = torch.from_numpy(getLogMelSpec(audio_path = f"{self.root_dir}/{self.filelist[index]}"))

    
    
    return data, data


class specEncoder(nn.Module):

  def __init__(self):
    super (specEncoder, self).__init__()

    #define layers
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
    self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
    self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 2, padding = 1)
    self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, stride = 2, padding = 1)
    self.conv5 = nn.Conv2d(in_channels = 8, out_channels = 4, kernel_size = 3, stride = 2, padding = 1)
    self.conv6 = nn.Conv2d(in_channels = 4, out_channels = 2, kernel_size = 3, stride = 2, padding = 1)
    self.conv7 = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = 3, stride = 2, padding = 1)

    ###### define attention conv layers

    self.att1 = nn.MaxPool2d(2, stride=2)
  
  def attention(self, input):
    return self.att1(self.att1(self.att1(self.att1(self.att1(self.att1(input)))))).max()*input
  
  def forward(self, x):
    #print(x.shape)
    self.out = self.conv1(x)
    #self.out = self.attention(self.out)
    self.out = self.conv2(self.out)
    self.out = self.conv3(self.out)
    self.out = self.conv4(self.out)
    self.out = self.conv5(self.out)
    self.out = self.conv6(self.out)
    self.out = self.conv7(self.out)
    
    return self.out


class specDecoder(nn.Module):

  def __init__(self):
    super (specDecoder, self).__init__()

    #define layers
    self.conv1 = nn.ConvTranspose2d(in_channels = 1, out_channels = 2, kernel_size = 2, stride = 2, padding = 0)
    self.conv2 = nn.ConvTranspose2d(in_channels = 2, out_channels = 4, kernel_size = 2, stride = 2, padding = 0)
    self.conv3 = nn.ConvTranspose2d(in_channels = 4, out_channels = 8, kernel_size = 2, stride = 2, padding = 0)
    self.conv4 = nn.ConvTranspose2d(in_channels = 8, out_channels = 16, kernel_size = 2, stride = 2, padding = 0)
    self.conv5 = nn.ConvTranspose2d(in_channels = 16, out_channels = 32, kernel_size = 2, stride = 2, padding = 0)
    self.conv6 = nn.ConvTranspose2d(in_channels = 32, out_channels = 64, kernel_size = 2, stride = 2, padding = 0)
    self.conv7 = nn.ConvTranspose2d(in_channels = 64, out_channels = 1, kernel_size = 2, stride = 2, padding = 0)
    
    
    ###### define attention conv layers

    self.att1 = nn.MaxPool2d(2, stride=2)
  
  def attention(self, input):
    return self.att1(self.att1(self.att1(self.att1(self.att1(self.att1(input)))))).max()*input
  
  def forward(self, x):
    self.out = self.conv1(x)
    self.out = self.conv2(self.out)
    self.out = self.conv3(self.out)
    self.out = self.conv4(self.out)
    self.out = self.conv5(self.out)
    self.out = self.conv6(self.out)
    self.out = self.out[:, :, :, 0:349]
    self.out = self.attention(self.out)
    self.out = self.conv7(self.out)
    
    return self.out




# define the LightningModule
class MusicGenerator(pl.LightningModule):
    def __init__(self, lr, batch_size, train_dir, val_dir, sampling_rate = 22050):
        super().__init__()
        self.encoder = specEncoder()
        self.decoder = specDecoder()
        self.learning_rate = lr
        self.batch_size = batch_size
        self.sr = sampling_rate
        self.TRAIN_DIR = train_dir
        self.VAL_DIR = val_dir


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        y = x
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        y = x
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        if batch_idx == 2:
            print(z[0])
            plt.figure(figsize=(14, 5))
            librosa.display.specshow(x[0,0,:,:].cpu().detach().numpy(), sr=self.sr, x_axis='time', y_axis='log')
            plt.colorbar()
            plt.show()

            plt.figure(figsize=(14, 5))
            librosa.display.specshow(x_hat[0,0,:,:].cpu().detach().numpy(), sr=self.sr, x_axis='time', y_axis='log')
            plt.colorbar()
            plt.show()
        # Logging to TensorBoard by default
        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        dataset = musicDataloader(root_dir = self.TRAIN_DIR)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, num_workers = 2)
        return dataloader

    def val_dataloader(self):
        dataset = musicDataloader(root_dir = self.VAL_DIR)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = False, num_workers = 2)
        return dataloader

