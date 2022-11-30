---
title: Home
layout: home
---

[Pix2Pix]를 활용하여 흑백 이미지를 컬러 이미지로 변환하는 튜토리얼에 오신 걸 환영합니다.  
Pix2Pix는 "Image-to-Image Translation with Conditional Adversarial Networks"에서 소개된 딥러닝 모델인데, 나온 지 오래되었지만 GAN의 역사에서 빼 놓을 수 없을 정도로 임팩트가 대단했고, 코드 또한 단순해서 GAN에 입문하는 사람들은 꼭 한번씩 읽어야만 하는 논문입니다. 이번 튜토리얼에서는 Pix2Pix를 데이터를 다운받는 것부터 시작해서, 모델 설계까지 딥러닝 개발의 모든 단계를 하나 하나 코딩할 것입니다.  

우선, 코드를 돌리는데 필요한 라이브러리들을 import합니다.
<script src="https://gist.github.com/youngjaeoh/89b7c8b7914a92e063f6da60779398a9.js"></script>



----

[^1]: [It can take up to 10 minutes for changes to your site to publish after you push the changes to GitHub](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll#creating-your-site).

[Pix2Pix]: https://arxiv.org/pdf/1611.07004.pdf
[Just the Docs]: https://just-the-docs.github.io/just-the-docs/
[GitHub Pages]: https://docs.github.com/en/pages
[README]: https://github.com/just-the-docs/just-the-docs-template/blob/main/README.md
[Jekyll]: https://jekyllrb.com
[GitHub Pages / Actions workflow]: https://github.blog/changelog/2022-07-27-github-pages-custom-github-actions-workflows-beta/
[use this template]: https://github.com/just-the-docs/just-the-docs-template/generate
