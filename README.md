# test-pytorch-lightning

## 環境構築

### install PyTorch (for Apple Silicon)

2021年5月4日現在、Apple Silicon 用パッケージをインストールする場合、
[wheel ファイル](https://github.com/wizyoung/AppleSiliconSelfBuilds/blob/main/builds/torch-1.8.0a0-cp39-cp39-macosx_11_0_arm64.whl)
を利用する。ダウンロード後、下記コマンドを実行する

```shell
$ pip install /path/to/torch-1.8.0a0-cp39-cp39-macosx_11_0_arm64.whl
```

### install PyTorch-Lightning

2021年5月4日現在、pip で [gRPC Python](https://github.com/grpc/grpc/tree/master/src/python/grpcio) をインストールする場合、 
パッケージを requirements.txt に記述し、下記コマンドを実行する

```shell
$ GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 pip install -r requirements.txt
```

## ログの確認

```shell
$ tensorboard --logdir ./lightning_logs
```
