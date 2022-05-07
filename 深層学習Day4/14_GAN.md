<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# GAN（Generative Adversarial Nets）
- 生成器と識別器を競わせて学習する生成&識別モデル
  - nGenerator:乱数からデータを生成
  - nDiscriminator: 入力データが真データ(学習データ)であるかを識別
- 2人のプレイヤーがミニマックスゲームを繰り返すと考えることができる。
  - GANでは価値関数𝑉に対し, 𝐷が最大化, 𝐺が最小化を行う
  - \\\(\displaystyle \min_g \max_d V(G,D)\\\)
  - \\\(V(D,G) = \mathbb{E}_{x \sim P_{data}(x)} \left[ logD(x)\right] + \mathbb{E}_{x \sim P_{z}(Z)} \left[ log\left(1 - D(G(z))\right)\right]\\\))
  - GANの価値関数はバイナリークロスエントロピーと考えることができる。
- 最適化方法
  - Discriminatorの更新
    - Generatorパラメータ\\\(\theta_g\\\)を固定
    - 真データと生成データを𝑚個ずつサンプル
    - \\\(\theta_d\\\)を勾配上昇法(Gradient Ascent)で更新
    - \\\(\displaystyle \frac{\partial}{\partial \theta_d}\frac{1}{m}[ log[ D(x) ]+log[ 1−D(G(z) ]]\\\)
    - \\\(\theta_d\\\)をk回更新
  - Generatorの更新
    - Discriminatorパラメータ\\\(\theta_d\\\)を固定
    - 生成データを𝑚個ずつサンプル
    - \\\(\theta_g\\\)を勾配降下法(Gradient Descent)で更新
    - \\\(\displaystyle \frac{\partial}{\partial \theta_g}\frac{1}{m}[ log[ 1−D(G(z) ]]\\\)
    - \\\(\theta_g\\\)を1回更新。
- なぜGeneratorは本物のようなデータを生成するのか︖7Ø生成データが本物とそっくりな状況とはn𝑝+=𝑝,-.-であるはずØ価値関数が𝑝)=𝑝*+,+の時に最適化されていることを示せばよい
- 二つのステップにより確認する
  - 1.𝐺を固定し、価値観数が最大値を取るときの𝐷𝒙を算出
  - 2.上記の𝐷𝒙を価値関数に代入し、𝐺が価値観数を最小化する条件を算出


# DCGAN
# Conditional GAN