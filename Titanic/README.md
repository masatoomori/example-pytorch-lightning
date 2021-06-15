# Titanic by PyTorch Lightning

## ディレクトリ構成

```text
├── README.md
├── data_collection (input/original のデータを作成するスクリプト)
├── data_processing
│   └── transform_data.py (input/original -> input/preprocessed を行うスクリプト)
├── input
│   ├── download (外部から入手した生データ。手作業でダウンロードするもの)
│   │   ├── test.csv
│   │   └── train.csv
│   ├── original (data_collection のスクリプトで生成されるデータ保管場所。スクリプトによるダウンロードデータ含む)
│   │   ├── modeling.csv
│   │   └── submission.csv
│   └── preprocessed (transform_data.py で加工した結果。main.py で読み込み)
│       ├── data_profile.json
│       ├── modeling.csv
│       ├── modeling.pkl
│       ├── submission.csv
│       └── submission.pkl
├── lightning_files
│   └── input
│       └── processed (main.py で加工した結果)
│           ├── modeling.pkl (機械学習で使うデータ。検証用データも含む)
│           └── submission.pkl (目的変数がないデータ。Kaggle 提出用)
├── lightning_logs
├── modeling
│   ├── model_evaluation.py
│   └── net_encoder_decoder_titanic.py
└── output
    ├── data
    └── model
        └── model.pth
```
