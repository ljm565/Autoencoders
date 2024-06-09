# Training Autoencoder
여기서는 autoencoder 모델을 학습하는 가이드를 제공합니다.

### 1. Configuration Preparation
Autoencoder 모델을 학습하기 위해서는 Configuration을 작성하여야 합니다.
Configuration에 대한 option들의 자세한 설명 및 예시는 다음과 같습니다.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: cpu                 # You can DDP training with multiple gpus. e.g. gpu: [0], [0,1], [1,2,3], cpu: cpu

# project config
project: outputs/AE         # Project directory
name: MNIST                 # Trained model-related data are saved at {$project}/{$name} folde

# model config
model_type: AE              # [AE, CAE], you can choose one.
denoising: False            # if True, denoising autoencder will be trained
noise_mean: 0               # if denoising, this parameter will be used
noise_std: 0.1              # if denoising, this parameter will be used

# image setting config
height: 28                  # Image size for preprocessing
width: 28                   # Image size for preprocessing
color_channel: 1            # [1, 3], you can choose one
convert2grayscale: False    # if True and color_channel is 3, you can train color image with grayscaled image

# data config
workers: 0                  # Don't worry to set worker. The number of workers will be set automatically according to the batch size.
MNIST_train: True           # if True, MNIST will be loaded automatically.
MNIST:
    path: data/
    MNIST_valset_proportion: 0.2      # MNIST has only train and test data. Thus, part of the training data is used as a validation set.
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null

# train config
batch_size: 128
epochs: 10
lr: 0.001
hidden_dim: 256
latent_dim: 32              # dimension of latent space.
dropout: 0.1

# logging config
common: ['train_loss', 'validation_loss']

# t-sne visualiztion config
result_num: 10
```

### 2. Training
#### 2.1 Arguments
`src/run/train.py`를 실행시키기 위한 몇 가지 argument가 있습니다.
* [-c, --config]: 학습 수행을 위한 config file 경로.
* [-m, --mode]: [`train`, `resume`] 중 하나를 선택.
* [-r, --resume_model_dir]: mode가 `resume`일 때 모델 경로. `{$project}/{$name}`까지의 경로만 입력하면, 자동으로 `{$project}/{$name}/weights/`의 모델을 선택하여 resume을 수행.
* [-l, --load_model_type]: [`loss`, `last`] 중 하나를 선택.
    * `loss`(default): Valdiation loss가 최소일 때 모델을 resume.
    * `last`: Last epoch에 저장된 모델을 resume.
* [-p, --port]: (default: `10001`) DDP 학습 시 NCCL port.


#### 2.2 Command
`src/run/train.py` 파일로 다음과 같은 명령어를 통해 autoencoder 모델을 학습합니다.
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir {$project}/{$name}
```
모델 학습이 끝나면 `{$project}/{$name}/weights`에 체크포인트가 저장되며, `{$project}/{$name}/args.yaml`에 학습 config가 저장됩니다.