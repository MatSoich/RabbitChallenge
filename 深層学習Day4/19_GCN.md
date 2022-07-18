<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

19 GCN
=======

# GCN
- CNNとGCNの違い
  - CNNの技術をグラフ構造に応用したのがGCN
    - 距離の代わりに、関係性の間に幾つのノードを経由するかで判断する。直接の繋がりが１、ノードを１つ経由すると2、など。
- GCNの応用分野
  - 新型コロナの感染予測
- 試験のポイント
  1. Spatial GCNとSpectral GCNの特徴
     - Speatial GCN
       - グラフにクラスタリングを繰り返す手法
       - 回数を繰り返すことでカバーできるノード同士の関係を広げていく。
       - （画像）
     - Spectral GCN
       - フーリエドメインでの畳み込みの特性を利用している。
       - グラフの調和解析が利用できる演算子グラフラプラシアンによって対応するベクトルの組み合わせを見つけ畳み込みをグラフ全体まで拡張させる。
       - グラフフーリエ変換とも呼ばれる
  2. GCNの弱点
     - Spatial GCNの弱点
       - 次元が低く、近傍がある場所が限られる場合、広い範囲で重みが持たせにくい
     - Spectral GCNの弱点
       - 計算量が多い。
       - グラフ間でサイズと構造が変わると使い回しができなくなる。
         - のちにこの弱点を克服するためにChebnetが登場。
         - GCNが空間領域に踏み込むきっかけになっている。
