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
코드를 보면 데이터를 다운로드하는 코드는 딱 1줄로, 상당히 간편한 것을 알 수 있습니다. untar_data(URLs.COCO_SMPLE)을 적어만 주면 데이터셋을 다운로드 후 자동으로 압축 해제까지 진행해 줍니다. 두 번째 줄은 glob 라이브러리를 사용해서 모든 이미지들의 경로를 리스트 형태로 묶어 주는데요, '/train_sample'는 이미지가 들어 있는 폴더 이름이고,  '/*.jpg'는 모든 jpg 확장자를 가진 파일들을 불러 오겠다는 의미입니다.  
이렇게 만들어진 path 변수에는 다운받은 모든 COCO 데이터셋의 이미지들에 대한 경로가 적혀지게 됩니다!



----
[Pix2Pix]: https://arxiv.org/pdf/1611.07004.pdf
