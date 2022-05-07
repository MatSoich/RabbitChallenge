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
   - CNN4にペア画像(それぞれの画像のデータをx1、x2とする)を入力し、その出力である特徴ベクトルをf(x1)、f(x2)とします。そして埋め込み空間内での2つの特徴ベクトルの距離D(f(x1),f(x2))を学習させる。
   - ペア画像が同じクラスの場合には距離Dが小さくなるように、逆に異なるクラスの場合には大きくなるように損失関数Lを設計し、学習を行う。
   - 損失関数は以下のように表す。Contrastive Lossと呼ばれる。
     - \\\(\displaystyle L = \frac{1}{2}\left[ yD^2 - (1 - y) max (m - D ,0)^2\right]\\\)
     - Dは元論文ではユークリッド距離が使われる。
     - m はマージン
     - yはx1およびx2の距離が同じ場合を表す。
     - 
   - 損失関数には大きな欠点があります。それは、異なるクラスのデータに対しては距離Dがマージンmを超えた時点で最適化が終了するのに対し、同じクラスのデータに対してはD=0となるまで、つまり埋め込み空間中のある1点に埋め込まれるまで最適化し続けてしまう点です。
2. Triplet Network
   - Triplet netwprkはSiamesenetworkの欠点を改良した手法であると主張されています。
     - 先ず、Siamese networkで問題となった同じクラスのペアと異なるクラスのペアの間に生じる不均衡は、Triplet networkでは解消されています。
     - Triplet networkでは同じクラスのデータの距離Dpを0にする必要はありません。つまりDpとDnそれぞれに対する条件をなくし、あくまでDpとDnを相対的に最適化することで、Siamese networkで問題となった不均衡を解消しています。
     - さらに、Triplet networkにはSiamese networkと比べ、学習時のコンテキストに関する制約が緩和される利点がある。
       - triplet lossの場合には、基準となる画像(アンカーサンプル)に対して類似度が低いサンプルをもう一方のサンプルよりも遠ざけることになるため、コンテキストを考慮する必要はありません
- Triplet Networkの問題点
  1. 学習がすぐに停滞してしまう。
     - 膨大な入力セットの組み合わせ(triplet)の内、学習に有効な入力セットを厳選する必要があります。このような入力セットの厳選の操作のことをtripletselectionやtriplet miningと言います。学習に有効な入力セットとは、triplet lossが発生するようなできるだけ「難しい入力セット」を意味します。
  2. クラス内距離がクラス間距離より小さくなることを保証しない
     - この問題を解決するために、入力セットを3つのデータから構成するのではなく、4つのデータで構成するQuadrupt lossと呼ばれる次の損失関数が提案されていま

- 