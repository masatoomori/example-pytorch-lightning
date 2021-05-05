import os
import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback

from net_encoder_decoder_mnist import Encoder, Decoder

LIGHTNING_PATH = os.path.join(os.curdir, 'lightning_files')
RESULT_PATH = os.path.join(os.curdir, 'result')
MODEL_FILE = os.path.join(RESULT_PATH, 'model.pt')
CHECKPOINT_FILE = os.path.join(RESULT_PATH, 'checkpoint.ckpt')

NUM_WORKERS = min(2, os.cpu_count())
os.makedirs(RESULT_PATH, exist_ok=True)

VALIDATION_RATIO = 0.2


# functions to show an image
def show_img(img, file='', text_=''):
    img = img / 2 + 0.5             # unnormalize
    npimg = img.detach().numpy()    # img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(x=3, y=2, s=text_, c="red")

    # plt.imshow(npimg)
    plt.pause(3)
    if file != '':
        plt.savefig(os.path.join(RESULT_PATH, '{}.png'.format(file)))
    plt.close()


class MyPrintingCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        print('')


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, data_dir=LIGHTNING_PATH):
        super().__init__()
        self.data_dir = data_dir

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        self.encoder = Encoder()    # nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 32))
        self.decoder = Decoder()    # nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):        # train, val, testデータ分割
        # Assign train/val datasets for use in dataloaders
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        n_val = int(len(mnist_full) * VALIDATION_RATIO)
        n_train = len(mnist_full) - n_val
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_full, [n_train, n_val])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        self.trainloader = DataLoader(self.mnist_train, shuffle=True, drop_last=True, batch_size=32,
                                      num_workers=NUM_WORKERS)
        # get some random training images
        return self.trainloader

    def val_dataloader(self):
        return DataLoader(self.mnist_val, shuffle=False, batch_size=32, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        self.testloader = DataLoader(self.mnist_test, shuffle=False, batch_size=32, num_workers=NUM_WORKERS)
        return self.testloader


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_gpu = 0 if device == torch.device('cpu') else 1

    autoencoder = LitAutoEncoder()

    # trainer = pl.Trainer()
    trainer = pl.Trainer(max_epochs=10, gpus=n_gpu, callbacks=[MyPrintingCallback()])
    trainer.fit(autoencoder)    # , DataLoader(train), DataLoader(val))
    print('training_finished')

    dataiter = iter(autoencoder.trainloader)
    images, labels = dataiter.next()
    # show images
    show_img(torchvision.utils.make_grid(images), 'mnist_initial', text_='original')
    # print labels
    print(' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
    results = trainer.test(autoencoder)
    print(results)

    dataiter = iter(autoencoder.testloader)
    images, labels = dataiter.next()
    # show images
    show_img(torchvision.utils.make_grid(images), 'mnist_results',
             text_=' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
    # print labels
    print(' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))

    # torchscript
    torch.jit.save(autoencoder.to_torchscript(), MODEL_FILE)
    trainer.save_checkpoint(CHECKPOINT_FILE)

    pretrained_model = autoencoder.load_from_checkpoint(CHECKPOINT_FILE)
    pretrained_model.freeze()
    pretrained_model.eval()

    latent_dim, ver = 32, 10
    dataiter = iter(autoencoder.testloader)
    images, labels = dataiter.next()
    # show images
    show_img(torchvision.utils.make_grid(images), 'original_images_{}_{}'.format(latent_dim, ver),
             text_=' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(8)))

    encode_img = pretrained_model.encoder(images[0:32].to('cpu').reshape(32, 28 * 28))
    decode_img = pretrained_model.decoder(encode_img)
    show_img(torchvision.utils.make_grid(decode_img.cpu().reshape(32, 1, 28, 28)),
             'original_autoencode_preds_mnist_{}_{}'.format(latent_dim, ver),
             text_=' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(8)))


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))
