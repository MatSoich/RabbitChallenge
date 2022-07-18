<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>


6 EfficientNet
==========
# EfficientNet
- AlexNet以降、CNNを大規模にすることで制度を改善するアプローチが主流
  - 例えば、RESNETはRESNET18からRESNET200
- 幅、深さ、画像の解像度をスケールアップすることで精度は向上したもののモデルが複雑になった
- EfficientNetモデル群は効率的なスケールアップの規則を採用することで、精度を高めると同時にパラメータ数を大幅に減少させた。
- 具体的にはモデルスケーリングの法則として、複合係数（Compound Coefficient）を導入して幅、深さ、画像の解像度の最適化を実施。
  - 詳細は論文を参照。（https://arxiv.org/abs/1905.11946）
- 精度はResnet50に比べてEfficientNet-B4は同程度の処理速度と計算量で精度が6.4%改善。
- 転移学習にも有効。

## Compund Scaringの詳細
- 幅(w)、深さ(d)、画像の解像度(r)はある程度まで増やすと精度は横這いになる。
- 一方で畳み込み演算のFLOPSは\\\(d, w^2, h^2\\\)に比例して増加。
- Compound Scaling Methodとして最適化問題を解くことで最適なd, w, rを導くことができる。
