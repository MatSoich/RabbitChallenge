<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# GRU
![kakunin](imgs/GRU.png)
- LSTMの改良版
  - LSTMは複雑すぎて学習に時間がかかる
  - パラメータを減らさず精度を保持しようというのが、GRU
  - 隠れ層h(t)に計算状態を保存するのがGRUの特徴
  - リセットゲートは隠れ層をどのような状態で保持するのかを管理する。
  - 更新ゲートは今回の記憶と前回の記憶を元に今回の最終アウトプットを制御する。

# 確認テスト２５
- LSTMとCECが抱える課題についてそれぞれ簡潔に述べよ。
  - LSTMは入力ゲート、出力ゲート、忘却ゲート、CECと４つのパラメータがあって、計算量が多くなることが課題。
  - 原因としてはCECに学習機能がないため、他の３つの機能が必要になってしまっている。

# GRU続き
- tensorflowを使用した実装を見る。
  - 3_2を見る。

# 演習チャレンジ
![kakunin](imgs/EnshuChallange06.png)
- 正解は4

# 確認テスト２６
- LSTMとGRUの違いを簡潔に述べよ。
- LSTMは入力ゲート、出力ゲート、忘却ゲート、そしてCECと４つのパラメータがある。
- 一方で、GRUはリセットゲート、更新ゲートと２つのパラメータがある。
- GRUの方がパラメータが少なく計算量が少なく済むため、LSTMの処理速度の問題を解消している。



