<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

VQ-VAE
=========


# VQ-VAE

- VQ-VAEは、VAE (Variational AutoEncoder)の派生技術にあたる生成モデルです。
- 「自然界の様々な事物の特徴を捉えるには離散変数の方が適している」という発想から、潜在変数が離散値となるように学習が行われます。
- 従来のVAEで起こりやすいとされる\posteriorcollapse"の問題を回避し、高品質のデータを生成することが可能となります。

- 両者の最大の違いは、
- VAE
  - 潜在変数zがGauss分布に従うベクトルになるように学習を行う
- VQ-VAE
  - 潜在変数zが離散的な数値となるように学習を行うという点です。
- VQ-VAEでは、EncoderとDecoderの間にEncoderの出力を離散的な潜在変数に対応させる「ベクトル量子化処理(VQ: Vector Quantization)」が行われます