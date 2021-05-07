import os
import time
import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from net_encoder_decoder_titanic import Encoder, Decoder

# data_preprocessing.pyで作成したファイル
MODELING_ORIG_FILE = os.path.join(os.curdir, 'input', 'preprocessed', 'modeling.csv')
SUBMISSION_ORIG_FILE = os.path.join(os.curdir, 'input', 'preprocessed', 'submission.csv')

LIGHTNING_PATH = os.path.join(os.curdir, 'lightning_files')
DATA_PATH_PREFIX = os.path.join('input', 'processed')
MODELING_DATA_FILE = 'modeling.pkl'
SUBMISSION_DATA_FILE = 'submission.pkl'

RESULT_PATH = os.path.join(os.curdir, 'result')
MODEL_FILE = os.path.join(RESULT_PATH, 'model.pt')
CHECKPOINT_FILE = os.path.join(RESULT_PATH, 'checkpoint.ckpt')

N_CPU = min(2, os.cpu_count())
os.makedirs(RESULT_PATH, exist_ok=True)

TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2


class MyPrintingCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        print('')


class MyLitModule(pl.LightningModule):
    def __init__(self, data_dir=LIGHTNING_PATH):
        super().__init__()
        self.data_dir = data_dir

        # Hardcode some dataset specific attributes
        self.target = 'Survived'        # 1: Survived, 0: Dead
        self.num_classes = 2
        self.classes = (0, 1)
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
        df_modeling = pd.read_csv(MODELING_ORIG_FILE, encoding='utf8', dtype=object)
        df_submission = pd.read_csv(SUBMISSION_ORIG_FILE, encoding='utf8', dtype=object)

        # 必要に応じて型の変換とかを書く。Transformというクラスを作った方がいいかもしれない
        df_modeling[self.target] = pd.to_numeric(df_modeling[self.target]).astype(np.int32)
        df_submission.drop([self.target], axis=1, inplace=True)

        drop_cols = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Salutation']
        df_modeling.drop(drop_cols, axis=1, inplace=True)
        df_submission.drop(drop_cols, axis=1, inplace=True)

        df_modeling = df_modeling.astype(np.float32)
        df_submission = df_submission.astype(np.float32)

        data_path = os.path.join(self.data_dir, DATA_PATH_PREFIX)
        os.makedirs(data_path, exist_ok=True)
        df_modeling.to_pickle(os.path.join(data_path, MODELING_DATA_FILE))
        df_submission.to_pickle(os.path.join(data_path, SUBMISSION_DATA_FILE))

    def setup(self, stage=None):
        # train, val, testデータ分割

        data_path = os.path.join(self.data_dir, DATA_PATH_PREFIX)
        df_full = pd.read_pickle(os.path.join(data_path, MODELING_DATA_FILE))

        ts_full = torch.tensor(df_full.drop(self.target, axis=1).values)
        ts_label = torch.tensor(df_full[self.target].values)
        ds_full = TensorDataset(ts_full, ts_label)

        n_full = len(df_full)
        n_test = int(n_full * TEST_RATIO)
        n_modeling = n_full - n_test
        ds_modeling, self.ds_test = torch.utils.data.random_split(ds_full, [n_modeling, n_test])

        n_val = int(n_modeling * VALIDATION_RATIO)
        n_train = n_modeling - n_val
        self.ds_train, self.ds_val = torch.utils.data.random_split(ds_modeling, [n_train, n_val])

    def train_dataloader(self):
        self.trainloader = DataLoader(self.ds_train, shuffle=True, drop_last=True, batch_size=32, num_workers=N_CPU)
        # get some random training data
        return self.trainloader

    def val_dataloader(self):
        return DataLoader(self.ds_val, shuffle=False, batch_size=32, num_workers=N_CPU)

    def test_dataloader(self):
        self.testloader = DataLoader(self.ds_test, shuffle=False, batch_size=32, num_workers=N_CPU)
        return self.testloader


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_gpu = 0 if device == torch.device('cpu') else 1

    model = MyLitModule()
    # trainer = pl.Trainer()
    trainer = pl.Trainer(max_epochs=10, gpus=n_gpu, callbacks=[MyPrintingCallback()])
    trainer.fit(model)    # , DataLoader(train), DataLoader(val))
    print('training_finished')

    model.prepare_data()
    model.setup()
    model.train_dataloader()
    dataiter = iter(model.trainloader)
    print(dataiter.next())


    exit()

    dataiter = iter(autoencoder.trainloader)
    images, labels = dataiter.next()
    # print labels
    print(' '.join('%5s' % autoencoder.classes[labels[j]] for j in range(4)))
    results = trainer.test(autoencoder)
    print(results)

    dataiter = iter(autoencoder.testloader)
    images, labels = dataiter.next()
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

    encode_img = pretrained_model.encoder(images[0:32].to('cpu').reshape(32, 28 * 28))
    decode_img = pretrained_model.decoder(encode_img)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))
