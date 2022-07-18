<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

18 MAML
========
# MAML
- MAMLのコンセプト
  - 深層学習に必要なデータの数を削減する
    - かつ少ないデータによる過学習を発生させない仕組み。
  - 転移学習、ファインチューニング、MAMLの違い
    - 転移学習
      - 事前学習済みモデルの全結合層のみ新しいデータで更新する。
    -  ファインチューニング
       - 事前学習済みモデルを初期値として学習を進める。
    - MAML
      - まずタスク共通の重みを学習し、その後その重みを前提に、各タスクの重みをファインチューニング。
      - 更新された各重みをSGDなどでまとめて、元の共通重みを更新。
      - 上記２ステップを繰り返す。
- 課題
  - MAMLは計算量が多い
    - タスクごとの勾配計算と共通パラメータの勾配計算の2回が必要
    - 実用的にはInner loopのステップ数を大きくできない
  - 計算コストを削減する改良案(近似方法)
    - First-orderMAML: 2次以上の勾配を無視し計算コストを大幅低減
    - Reptile: Inner loopの逆伝搬を行わず、学習前後のパラメータの差を利用