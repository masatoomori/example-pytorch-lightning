import os
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset

HOME_PATH = os.path.join(os.pardir)
INPUT_PATH = os.path.join(HOME_PATH, 'input', 'preprocessed')
MODELING_DATA_FILE = 'modeling.pkl'
DATA_PROFILE_FILE = 'data_profile.json'

TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2


class MyTransform():
	def __init__(self) -> None:
		pass

	def __call__(self, label):
		return self.transform(label)

	def fit(self, labels):
		pass

	def transform(self, label):
		pass


class MyDataset(Dataset):
	def __init__(self,
				 data_path=INPUT_PATH,
				 data_profile_file=DATA_PROFILE_FILE,
				 modeling_data_file=MODELING_DATA_FILE,
				 test_ratio=TEST_RATIO,
				 validation_ratio=VALIDATION_RATIO,
				 transform=None):
		self.data_profile = json.load(open(os.path.join(data_path, data_profile_file)))
		self.target = self.data_profile['target']['name']
		self.num_classes = self.data_profile['target']['num_classes']
		self.classes = self.data_profile['target']['classes']
		self.dims = self.data_profile['explanatory']['dims']
		self.x_cols = self.data_profile['explanatory']['names']
		self.is_classification = True if self.data_profile['prediction_type'] == 'classification' else False  # False means regression
		self.label_dtype = torch.long if self.is_classification else torch.float32

		df_full = pd.read_pickle(os.path.join(data_path, modeling_data_file))
		ds_full = TensorDataset(
			torch.tensor(df_full.drop(self.target, axis=1).values, dtype=torch.float32),
			torch.tensor(df_full[self.target].values, dtype=self.label_dtype)
			)
		n_full = len(df_full)

		n_test = int(n_full * test_ratio)
		n_modeling = n_full - n_test
		ds_modeling, self.ds_test = torch.utils.data.random_split(ds_full, [n_modeling, n_test])

		n_val = int(n_modeling * validation_ratio)
		n_train = n_modeling - n_val
		self.ds_train, self.ds_val = torch.utils.data.random_split(ds_modeling, [n_train, n_val])

		self.transform = transform


def main():
	dataset = MyDataset()


def test():
	main()


if __name__ == '__main__':
	test()
