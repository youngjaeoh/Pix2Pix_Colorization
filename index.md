---
title: Home
layout: home
---

[Pix2Pix]를 활용하여 흑백 이미지를 컬러 이미지로 변환하는 튜토리얼에 오신 걸 환영합니다.  
Pix2Pix는 "Image-to-Image Translation with Conditional Adversarial Networks"에서 소개된 딥러닝 모델인데, 나온 지 오래되었지만 GAN의 역사에서 빼 놓을 수 없을 정도로 임팩트가 대단했고, 코드 또한 단순해서 GAN에 입문하는 사람들은 꼭 한번씩 읽어야만 하는 논문입니다. 이번 튜토리얼에서는 Pix2Pix를 데이터를 다운받는 것부터 시작해서, 모델 설계까지 딥러닝 개발의 모든 단계를 하나 하나 코딩할 것입니다.  

우선, 코드를 돌리는데 필요한 라이브러리들을 import합니다.
<script src="https://gist.github.com/youngjaeoh/89b7c8b7914a92e063f6da60779398a9.js"></script>

대부분의 라이브러리들은 익숙하지만, skimage의 rgb2lab, lab2rgb 그리고 fastai 라이브러리는 딥러닝 개발 시에 자주 접하는 라이브러리는 아니죠. skimage 라이브러리는 이후에 설명하도록 하고, 우선 fastai 라이브러리를 import 하는 이유는 데이터셋을 쉽게 다운 받을 수 있기 때문입니다. fastai 라이브러리를 활용하여 데이터를 다운 받는 코드를 보시죠.
<script src="https://gist.github.com/youngjaeoh/00675769dc3a561e1ae434dea03d9b69.js"></script>
코드를 보면 데이터를 다운로드하는 코드는 딱 1줄로, 상당히 간편한 것을 알 수 있습니다. untar_data(URLs.COCO_SAMPLE)을 적어만 주면 데이터셋을 다운로드 후 자동으로 압축 해제까지 진행해 줍니다. 두 번째 줄은 glob 라이브러리를 사용해서 모든 이미지들의 경로를 리스트 형태로 묶어 주는데요, '/train_sample'는 이미지가 들어 있는 폴더 이름이고,  '/*.jpg'는 모든 jpg 확장자를 가진 파일들을 불러 오겠다는 의미입니다.  
이렇게 만들어진 path 변수에는 다운받은 모든 COCO 데이터셋의 이미지들에 대한 경로가 적혀지게 됩니다!  

다만, COCO_SAMPLE의 경우 이미지가 너무 많아서 훈련 시간이 너무 길어질 수 있으니, 전체 이미지 중에서 10,000장만 골라와서 사용해 봅시다!
<script src="https://gist.github.com/youngjaeoh/5aafa054158b22ba218a241c483485e2.js"></script>
데이터를 나누기 전, 언제나 데이터가 똑같이 나누어지도록 np.random.seed(1)을 활용하여 랜덤 시드를 설정해 줍니다. 시드 설정 후, np.random.choice(paths, 10000, replace=False) 함수를 통하여 앞서 저장한 path에 있는 모든 이미지 경로 중 10000개의 경로만 랜덤으로 선택하여 chosen_paths 변수에 저장합니다. 3번째 줄의 index의 경우, 0~9999의 수를 랜덤 순서의 리스트로 반환합니다. 예를 들면 [ 33, 5000, 9888, 0, 1, ...] 이런 식으로 말이죠.  
이렇게 생성된 chosen path와 index를 가지고 이제 train set과 validation set으로 데이터를 나누어 주어야 합니다. train_path는 chosen_paths[index[:8000]]으로 앞 8000장을 선택하고, val_path는 chosen_paths[index[8000:]]으로 설정하여 나머지 2000장을 선택하여 train set과 validation set의 비율을 8:2로 나누어 줍니다. 물론 굳이 8:2 비율을 고수하지 않으시고, 전체 데이터셋을 다 훈련에 활용하고, validation 셋은 따로 두고 사용하여도 무방합니다.  
현재까지 저희는 이미지를 다룬 게 아니라 "이미지의 경로"만 만지작거렸으니, 실제로 이미지가 잘 들어가 있는지 확인을 해 봐야겠죠? 아래의 코드를 활용하면 쉽게 이미지를 불러오고 display할 수 있습니다.
<script src="https://gist.github.com/youngjaeoh/78833412a664deff6ba250ccf67a3014.js"></script>
테스트 겸 train_path[0]으로 첫 번째 이미지를 불러온 다음 plt.imshow() 함수를 통해서 이미지를 display합니다. 세 번째 줄의 plt.axis('off')의 경우, 이미지를 그냥 그리게 된다면 x축과 y축에 숫자가 뜨는데, 이를 없애서 깔끔하게 이미지를 그려주는 역할을 합니다.  

이제 이미지들이 잘 들어가 있는 것을 확인하였으니, 이미지 전처리 클래스를 생성할 차례입니다. 코드를 보면서 단계별로 설명하겠습니다.
<script src="https://gist.github.com/youngjaeoh/b6135372152dfffa6d48f37f59123dff.js"></script>
전처리 클래스는 총 세개의 함수로 나누어져 있으며, 이들은 __init__(), __getitem__(), __len()__ 함수입니다.  
우선 __init__() 부터 확인해 보겠습니다. 이 함수는 mode가 train인지, validation인지 확인하여 이에 걸맞는 transform을 선언해 줍니다. 만약 train일 경우, transforms.Compose() 함수를 사용해서 이미지 크기를 256x256 사이즈로 맞추어 주고, 랜덤으로 좌우 반전 augmentation을 진행하는 변환 과정을 체이닝해 줍니다. validation의 경우 augmentation을 건너뛰고 이미지 resize만 진행합니다.  
__getitem__() 함수의 경우, index를 받아서 이미지를 불러온 다음, __init__에서 만들었던 transform에 넣어서 이미지를 변환한 이후,  skimage의 rgb2lab 라이브리를 활용해서 rgb 컬러 채널을 Lab 컬러 채널로 바꾸어 줍니다.  



----
[Pix2Pix]: https://arxiv.org/pdf/1611.07004.pdf
