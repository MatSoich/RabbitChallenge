<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# Metric-Learning
- ディープラーニング技術を用いた距離学習は、人物同定(Person Re-Identification)をはじめ、顔認識、画像分類、画像検索、異常検知など幅広いタスクに利用される
- 特徴ベクトルに意味を持たせるには何らかの手法でCNNを学習する必要があります。深層距離学習はその1つの方法
  - 同じクラスに属する(=類似)サンプルから得られる特徴ベクトルの距離は小さくする
  - 異なるクラスに属する(=非類似)サンプルから得られる特徴ベクトルの距離は大きくする
- 以下で、深層距離学習の代表的な手法であるSiamese networkとTripletnetworkについて説明していきます
1. Siamese network
   - 2006年に提案された深層距離学習の中でも比較的歴史の古い手法です。
   - Siamese networkの特徴は、2つのサンプルをペアで入力しそれらのサンプル間の距離を明示的に表現して調整する点