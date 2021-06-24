import argparse
import os
import time
import json
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from titanic_dataset import MyDataset
from net_encoder_decoder_titanic import Encoder, Decoder

HOME_PATH = os.pardir

# MyDatasetに必要な環境変数
INPUT_PATH = os.path.join(HOME_PATH, 'input', 'preprocessed')
MODELING_DATA_FILE = 'modeling.pkl'
DATA_PROFILE_FILE = 'data_profile.json'
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2

LIGHTNING_PATH = os.path.join(os.curdir, 'lightning_files')
DATA_PATH_PREFIX = os.path.join('input', 'processed')

TEST_DATA_FILE = 'submission.pkl'

RESULT_PATH = os.path.join(HOME_PATH, 'output')
MODEL_RESULT_PATH = os.path.join(RESULT_PATH, 'model')
DATA_RESULT_PATH = os.path.join(RESULT_PATH, 'data')
MODEL_FILE = os.path.join(MODEL_RESULT_PATH, 'model.pth')
FULL_RESULT_FILE = os.path.join(DATA_RESULT_PATH, 'full_result.csv')
PREDICTION_RESULT_FILE = os.path.join(DATA_RESULT_PATH, 'prediction_result.csv')

os.makedirs(MODEL_RESULT_PATH, exist_ok=True)
os.makedirs(DATA_RESULT_PATH, exist_ok=True)

N_CPU = min(2, os.cpu_count())

MAX_EPOCH = 2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32


class MyPrintingCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        print('')


class MyLitModule(pl.LightningModule):
    def __init__(self, data_dir=LIGHTNING_PATH, dataset=None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset

        self.data_profile = dataset.data_profile
        self.num_classes = dataset.data_profile['target']['num_classes']
        self.dims = dataset.data_profile['explanatory']['dims']
        self.encoder = Encoder(self.dims, self.num_classes)
        self.decoder = Decoder(self.dims, self.num_classes)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self.encoder(x).squeeze()

        if self.num_classes == 1:
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(y_hat.float(), y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.encoder(x).squeeze()

        if self.num_classes == 1:
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(y_hat.float(), y.float())
        self.log('test_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.ds_train = self.dataset.ds_train
        self.ds_val = self.dataset.ds_val
        self.ds_test = self.dataset.ds_test
        self.x_cols = self.dataset.x_cols
        self.target = self.dataset.target

    def train_dataloader(self, shuffle=True, batch_size=BATCH_SIZE):
        # get some random training data
        self.trainloader = DataLoader(self.ds_train, shuffle=shuffle, drop_last=True, batch_size=batch_size, num_workers=N_CPU)
        return self.trainloader

    def val_dataloader(self, batch_size=BATCH_SIZE):
        return DataLoader(self.ds_val, shuffle=False, batch_size=batch_size, num_workers=N_CPU)

    def test_dataloader(self):
        return DataLoader(self.ds_test, shuffle=False, batch_size=len(self.ds_test), num_workers=N_CPU)


def get_result_from_model(model):
    data = DataLoader(ds, batch_size=len(ds), shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                      collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)

    dataiter = iter(data)
    explanatory_values, labels = dataiter.next()
    df = pd.DataFrame(explanatory_values.numpy(), columns=model.x_cols)
    df[model.target] = labels.numpy()

    y_hat = model.encoder(explanatory_values).squeeze().detach().numpy()
    df[model.target + '_pred'] = np.where(y_hat > 0.1, 1, 0)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', action='store_true')
    parser.add_argument('--discard_model', '-d', action='store_true')
    parser.add_argument('--evaluate', '-e', action='store_true')
    parser.add_argument('--predict', '-p', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_gpu = 0 if device == torch.device('cpu') else 1

    dataset = MyDataset(
        data_path=INPUT_PATH,
        data_profile_file=DATA_PROFILE_FILE,
        modeling_data_file=MODELING_DATA_FILE,
        test_ratio=TEST_RATIO,
        validation_ratio=VALIDATION_RATIO
    )
    model = MyLitModule(dataset=dataset)

    if args.train:
        trainer = pl.Trainer(max_epochs=MAX_EPOCH, gpus=n_gpu, callbacks=[MyPrintingCallback()])
        trainer.fit(model)  # , DataLoader(train), DataLoader(val))
        print('training_finished')

        dataiter = iter(model.test_dataloader())
        explanatory_values, labels = dataiter.next()
        results = trainer.test(model)
        print(explanatory_values[:5])
        print(labels[:5])
        print(results)

        if args.discard_model:
            print('trained model discarded')
        else:
            torch.save(model.state_dict(), MODEL_FILE)
            print('trained model saved')
        print("'$ tensorboard --logdir ./lightning_logs' to check result")

    if args.evaluate:
        # load model
        if args.train:
            print('evaluate trained model')
        else:
            model = MyLitModule(dataset=dataset)
            model.setup()
            model.load_state_dict(torch.load(MODEL_FILE))
            print('evaluate loaded model')

        model.eval()
        model.freeze()

        # train data
        print()



        # load data
        df_train = get_result_from_model(model.ds_train, model)
        df_val = get_result_from_model(model.ds_val, model)
        df_test = get_result_from_model(model.ds_test, model)

        df_train['data_usage'] = 'train'
        df_val['data_usage'] = 'val'
        df_test['data_usage'] = 'test'

        df_full = pd.concat([df_train, df_val, df_test], sort=False)
        df_full.to_csv(FULL_RESULT_FILE, index=False)

    if args.predict:
        # load model
        if args.train:
            print('evaluate trained model')
        else:
            model = MyLitModule()
            model.setup()
            model.load_state_dict(torch.load(MODEL_FILE))
            print('evaluate loaded model')

        # load data
        df_submission = pd.read_pickle(os.path.join(LIGHTNING_PATH, DATA_PATH_PREFIX, SUBMISSION_DATA_FILE))
        df_submission[DATA_PROFILE['target']['name']] = None        # 形を整える為にカラムを追加する
        X = torch.tensor(df_submission.drop(DATA_PROFILE['target']['name'], axis=1).values, dtype=torch.float32)
        y_hat = model.encoder(X).squeeze().detach().numpy()
        df_submission[DATA_PROFILE['target']['name']] = y_hat
        df_submission.to_csv(PREDICTION_RESULT_FILE, index=False)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))
