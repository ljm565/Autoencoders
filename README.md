# Autoencoders
## 설명
Autoencoder (오토인코더)는 원래의 데이터를 압축하고 복구하는 과정에서, 의미있는 잠재 변수를 추출하여 데이터를 압축하거나 차원을 축소하는 manifold learning을 위한 모델입니다. 본 코드에서는 세 종류의 autoencoder 코드를 확인할 수 있으며, MNIST 데이터의 경우 학습 결과로 나온 잠재 변수(latent variable)를 t-SNE를 통해 2차원 데이터로 가시화 합니다.
<br><br><br>

## 모델 종류
* ### Vanilla Autoencoder
    Linear layer를 사용한 vanilla autoencoder가 구현되어 있습니다.
    <br><br>

* ### Convolutional Autoencoder
    Convolutional layer를 사용한 convoluational autoencoder가 구현되어 있습니다. 이 모델은 조금 더 복잡한 데이터에 대해 성능을 높이고싶을 때 vanilla autoencoder 대신에 사용할 수 있습니다.
    <br><br>

* ### Denoising Autoencoder
    데이터에 noise를 주어 denoising autoencoder 모델을 학습할 수 있습니다. 이 기법을 위의 vanilla autoencoder, convolutional autencoder에 모두 적용할 수 있습니다. Denoising autoencoder는 데이터의 좀 더 의미있는 잠재 변수(latent variable)를 추출하기 위해 사용 가능합니다.
    <br><br><br>

## 사용 데이터
* 실험으로 사용하는 데이터는 [Yann LeCun, Corinna Cortes의 MNIST](http://yann.lecun.com/exdb/mnist/) 데이터입니다.
* 학습 데이터의 경로를 설정하여 사용자가 가지고 있는 데이터도 학습 가능합니다.
<br><br><br>


## 사용 방법
* ### 학습 방법
    학습을 시작하기 위한 argument는 4가지가 있습니다.<br>
    * [-d --device] {cpu, gpu}, **필수**: 학습을 cpu, gpu로 할건지 정하는 인자입니다.
    * [-m --mode] {train, test}, **필수**: 학습을 시작하려면 train, 학습된 모델을 가지고 있어서 가시화 혹은 결과를 보고싶은 경우에는 test로 설정해야합니다. test를 사용할 경우, -n 이 **필수**입니다.
    * [-c --cont] {1}, **선택**: 학습이 중간에 종료가 된 경우 다시 저장된 모델의 체크포인트 부분부터 학습을 시작할 수 있습니다. 이 인자를 사용할 경우 -m train 이어야 합니다. 
    * [-n --name] {name}, **선택**: 이 인자는 -c 1 혹은 -m test 경우 사용합니다.
    중간에 다시 불러서 학습을 할 경우 모델의 이름을 입력하고, test를 할 경우에도 test 할 모델의 이름을 입력해주어야합니다(최초 학습시 config.json에서 정한 모델의 이름의 폴더가 형성되고 그 폴더 내부에 모델 및 모델 파라미터가 json 파일로 형성 됩니다).<br><br>

    예시<br>
    * 최초 학습 시: python3 main.py -d cpu -m train
    * 중간에 중단 된 모델 이어서 학습 시: python3 main.py -d gpu -m train -c 1 -n my_autoencoder
    * 학습 된 모델 결과 볼 때: python3 main.py -d cpu -m test -n my_autoencoder
    <br><br><br>

* ### 모델 학습 조건 설정 (config.json)
    * model_type: {AE, CAE} 중 선택, autoencoder의 기본 구조 선택(AE: vanilla autoencoder, CAE: convolutional autoencoder).
    * denoising: {0, 1} 중 선택, 1로 할 경우 model_type으로 지정한 모델에 대해 denoising autoencoder 모델 학습.
    * noise_mean, noise_std: denoising이 1일 경우 데이터에 줄 노이즈의 평균, 표준편차 값 설정.
    <br><br>

    * ### MNIST 데이터 사용 시
        * MNIST_train: {0, 1} 중 선택, 0인 경우 사용자 지정 데이터, 1인 경우 MNIST 데이터 학습.
        * MNIST_valset_proportion: **MNIST를 학습할 때 사용됨.** 학습 모델을 저장하기 위한 지표로 사용 될 valset을 trainset에 대해 설졍한 비율로 랜덤하게 선택(e.g. 0.2인 경우, 10,000개의 trainset 중 20 %를 valset, 나머지 80 %를 trainset으로 분리하여 학습).
        <br><br>
    
    * ### 사용자 지정 데이터 사용 시
        * 
        * 
        <br><br>

    * base_path: 학습 관련 파일이 저장될 위치
    * model_name: 학습 모델이 저장될 파일 이름 설정. **확장자는 .pt 로 작성.** base_path/model/{.pt 앞쪽 model_name}/ 내부에 model_name으로 모델 저장.
    * data_name: 학습 데이터 이름 설정. base_path/data/ 내부에 train.pkl, val.pkl, test.pkl 파일로 저장. 전처리 등 시간 절약을 위해 이후 같은 data_name을 사용할 시 저장된 데이터를 불러서 사용.
    * loss_data_name: 학습 시 발생한 loss data를 저장하기 위한 이름 설정. base_path/loss/ 내부에 저장.
    * color_channel: 학습에 사용되는 데이터가 흑백이면 1, 칼라면 3으로 설정(MNIST 사용 시 1로 설정).
    * hieght, width: 데이터의 전 처리 할 크기를 지정(MNIST의 raw data 크기는 28 * 28)
    * convert2grayscale: {0, 1} 중 선택, color_channel = 3 이고 흑백 데이터로 변경하고싶을 때만 사용, **이외의 경우에는 0으로 설정.**

## 작성중


    

<br><br><br>




## License
© 2022. Jun-Min Lee. All rights reserved.<br>
ljm56897@gmail.com, ljm565@kaist.ac.kr, ljm565@naver.com

