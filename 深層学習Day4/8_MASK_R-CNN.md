<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# MASK-RCNN
## R-CNN
- RCNN系のもとであるR-CNNはDeep Learningを使わない物体検知の手法
  1. 関心領域（ROI, Region of Interest）を切り出す。類似する領域をグルーピング
  2. 候補領域の画像の大きさを揃える。CNNにより特徴量を求める。
  3. CNNで求めた特徴量をSVMで学習。未知画像も特徴量をCNNで求め、学習ずみSVMで分類する。

## Fast R-CNN
- R-CNNの改良版
  - 関心領域ごとに畳み込み層に通すのではなく、画像ごとに一回の畳み込み操作を行う。
    - 計算量を大幅に削減。

## Faster R-CNN
- 関心領域の切り出しもCNNで行う。
- ほぼリアルタイムで動作し、動画認識への応用も可能になった。

# YOLO
- RCNN系以外の物体検知手法として、YOLO系統やSSD系統がある。
  - これらはRCNN系統と違い、1段階検出器である。
- YOLOの手法は以下のようになる
 1. 入力画像をグリッドに分割
 2. A. バウンディングボックスで候補領域を抽出すると同時に、物体なのか背景なのかを表す信頼度スコアを抽出。
 3. B.各グリッドでクラス分類を行う。
 4. AとBの情報を組み合わせて物体認識を行う。
- YOLOのメリット
  - 処理が相対的に早い。（一段階検出器）
  - 画像全体を見て予測することができるのでFast RCNNの半分以下の誤検出。

# インスタンスセグメンテーション
- セマンティックセグメンテーションとの違いは、同クラス内の複数の物体を区別できるか。
- 有名なアプローチとして、YOLACT,MASK-RCNNがある。
- YOLACTはYOLOのように一段階検出器となっている。

# MASK-RCNN
- Faster RCNNを拡張したインスタンスセグメンテーション用のアルゴリズム。
- MASK-RCNNでは物体検知した領域のみに検証を行uことで、効率化を実施することに成功している。
- ROI PoolingとROI Align
  - MASK-RCNNではより高度のROI Alignを使用。
- ROI Alignの手順
  1. N X Nの領域に分割（N X N　の特徴マップにするため）
  2. 領域の１つ１つについて４つの点を打つ。
  3. １つ１つの点について、周りの４つのピクセルを元に、何らかのルールで点の値を求める。
  4. ４つの点を１つにまとめる。