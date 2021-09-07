# BiLSTM Onset Detection

#### BiLSTM_Onsetnet 是一个基于双向LSTM和卷积神经网络（CRNN）的onset检测网络。目前我们将BiLSTM在视唱人声数据上进行训练和检测，并在一些公开数据集上进行了评价。

1. #### requirements 

```shell
conda install --yes --file requirements.txt
或者
pip install -r requirements.txt
```

2. #### 训练（train）

```shell
python train.py --cfg bi_lstm_87×267_adam_lr1e-5.yaml --model_type bi_lstm_onsetnet "WORKERS" 0 "DATASET.DIR" 'fold_0'
```

- #### `--cfg` 设置要求的配置文件 （bi_lstm_87×267_adam_lr1e-5.yaml）

- #### `--model_type` 设置所训练的模型 类型（bi_lstm_onsetnet）

- #### `WORKERS` 设置dataloader读取时使用的线程数

- #### `DATASET.DIR` 设置训练集和验证集的文件夹名

3. #### 测试（test）

```shell
python test.py --cfg home/data/ywm_data/Models/39to31.yaml --model bi_lstm_onsetnet --model_path home/data/ywm_data/bi_lstm_onsetnet_output/output/fold_0/bilstm/bi_lstm_39×267_adam_lr1e-5.fold_0/2021-06-09-09-04/best.pth "WORKERS" 4 "DATASET.DIR" 'kfold1'
```

- #### `--cfg` 设置要求的配置文件 （bi_lstm_87×267_adam_lr1e-5.yaml）

- #### `--model` 设置所测试的模型类型 （bi_lstm_onsetnet）

- #### `-model_path` 模型的路径

- #### `WORKERS` 设置dataloader读取时使用的线程数

- #### `DATASET.DIR` 任意

4. #### 性能测试结果

![image-20210907171625947](G:\labWork\GithubProject\bilstm_onset\performance.png)