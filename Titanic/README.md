# Titanic by PyTorch Lightning

## ディレクトリ構成

```text
├── README.md
├── transform_data.py
├── main.py
├── net_encoder_decoder_titanic.py
├── input
│    ├── original (外部から入手した生データ)
│    │    ├── test.csv
│    │    └── train.csv
│    └── preprocessed (transform_data.py で加工した結果。main.py で読み込み)
│         ├── modeling.csv
│         └── submission.csv
├── lightning_files
│    └── input
│         └── processed (main.py で加工した結果)
│             ├── modeling.pkl (機械学習で使うデータ。検証用データも含む)
│             └── submission.pkl (目的変数がないデータ。Kaggle 提出用)
└── result
```