import os
import time
import json
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from net_encoder_decoder_titanic import Encoder, Decoder

# data_preprocessing.pyで作成したファイル
MODELING_ORIG_FILE = os.path.join(os.curdir, 'input', 'preprocessed', 'modeling.pkl')
SUBMISSION_ORIG_FILE = os.path.join(os.curdir, 'input', 'preprocessed', 'submission.pkl')
DATA_PROFILE_FILE = os.path.join(os.curdir, 'input', 'preprocessed', 'data_profile.json')
DATA_PROFILE = json.load(open(DATA_PROFILE_FILE))

LIGHTNING_PATH = os.path.join(os.curdir, 'lightning_files')
DATA_PATH_PREFIX = os.path.join('input', 'processed')
MODELING_DATA_FILE = 'modeling.pkl'
SUBMISSION_DATA_FILE = 'submission.pkl'

RESULT_PATH = os.path.join(os.curdir, 'result')
MODEL_FILE = os.path.join(RESULT_PATH, 'model.pth')
TORCH_SCRIPT_FILE = os.path.join(RESULT_PATH, 'model.pt')

N_CPU = min(2, os.cpu_count())
os.makedirs(RESULT_PATH, exist_ok=True)

TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
MAX_EPOCH = 100
LEARNING_RATE = 1e-3


class MyPrintingCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        print('')


class MyLitModule(pl.LightningModule):
    def __init__(self, data_dir=LIGHTNING_PATH):
        super().__init__()
        self.data_dir = data_dir

        # Hardcode some dataset specific attributes
        self.target = DATA_PROFILE['target']['name']
        self.num_classes = DATA_PROFILE['target']['num_classes']
        self.classes = set(DATA_PROFILE['target']['classes'])
        self.dims = set(DATA_PROFILE['explanatory']['dims'])

        self.encoder = Encoder(self.dims,
                               self.num_classes)  # nn.Sequential(nn.Linear(, 128), nn.ReLU(), nn.Linear(128, ))
        self.decoder = Decoder(self.dims,
                               self.num_classes)  # nn.Sequential(nn.Linear(, 128), nn.ReLU(), nn.Linear(128, ))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self.encoder(x).squeeze()

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(y_hat.float(), y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.encoder(x).squeeze()

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(y_hat.float(), y.float())
        self.log('test_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def prepare_data(self):
        # download
        df_modeling = pd.read_pickle(MODELING_ORIG_FILE)
        df_submission = pd.read_pickle(SUBMISSION_ORIG_FILE)

        # 必要に応じて型の変換とかを書く。Transformというクラスを作った方がいいかもしれない
        # 基本的にはdata_preprocessing.pyで前処理は済ませたい

        data_path = os.path.join(self.data_dir, DATA_PATH_PREFIX)
        os.makedirs(data_path, exist_ok=True)
        df_modeling.to_pickle(os.path.join(data_path, MODELING_DATA_FILE))
        df_submission.to_pickle(os.path.join(data_path, SUBMISSION_DATA_FILE))

    def setup(self, stage=None):
        # train, val, testデータ分割

        data_path = os.path.join(self.data_dir, DATA_PATH_PREFIX)
        df_full = pd.read_pickle(os.path.join(data_path, MODELING_DATA_FILE))

        ts_full = torch.tensor(df_full.drop(self.target, axis=1).values, dtype=torch.float32)
        ts_label = torch.tensor(df_full[self.target].values, dtype=torch.long)
        ds_full = TensorDataset(ts_full, ts_label)

        n_full = len(df_full)
        n_test = int(n_full * TEST_RATIO)
        n_modeling = n_full - n_test
        ds_modeling, self.ds_test = torch.utils.data.random_split(ds_full, [n_modeling, n_test])

        n_val = int(n_modeling * VALIDATION_RATIO)
        n_train = n_modeling - n_val
        self.ds_train, self.ds_val = torch.utils.data.random_split(ds_modeling, [n_train, n_val])

    def train_dataloader(self):
        # get some random training data
        self.trainloader = DataLoader(self.ds_train, shuffle=True, drop_last=True, batch_size=32, num_workers=N_CPU)
        return self.trainloader

    def val_dataloader(self):
        return DataLoader(self.ds_val, shuffle=False, batch_size=32, num_workers=N_CPU)

    def test_dataloader(self):
        return DataLoader(self.ds_test, shuffle=False, batch_size=32, num_workers=N_CPU)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_gpu = 0 if device == torch.device('cpu') else 1

    model = MyLitModule()
    # trainer = pl.Trainer()
    trainer = pl.Trainer(max_epochs=MAX_EPOCH, gpus=n_gpu, callbacks=[MyPrintingCallback()])
    trainer.fit(model)  # , DataLoader(train), DataLoader(val))
    print('training_finished')

    # dataiter = iter(model.train_dataloader())
    # explanatory_values, labels = dataiter.next()
    # print(explanatory_values[:5])
    # print(labels[:5])

    dataiter = iter(model.test_dataloader())
    explanatory_values, labels = dataiter.next()
    results = trainer.test(model)
    print(explanatory_values[:5])
    print(labels[:5])
    print(results)

    # save model
    torch.save(model.state_dict(), MODEL_FILE)
    torch.jit.save(model.to_torchscript(), TORCH_SCRIPT_FILE)

    # load model
    model = MyLitModule()
    model.setup()
    model.load_state_dict(torch.load(MODEL_FILE))
    dataiter = iter(model.test_dataloader())
    explanatory_values, labels = dataiter.next()
    results = trainer.test(model)
    print(explanatory_values[:5])
    print(labels[:5])
    print(results)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))
