<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# MASK-RCNN
## R-CNN
- RCNN系のもとであるR-CNNはDeep Learningを使わない物体検知の手法
  1. 関心領域（ROI, Region of Interest）を切り出す。類似する領域をグルーピング
     - 物体候補領域の提案
  2. 候補領域の画像の大きさを揃える。CNNにより特徴量を求める。
  3. CNNで求めた特徴量をSVMで学習。未知画像も特徴量をCNNで求め、学習ずみSVMで分類する。
     - 提案された候補領域における物体のクラス分類

- R-CNNの課題は1および2-3の処理が遅いこと。
## Fast R-CNN
- R-CNNの改良版
  - 関心領域ごとに畳み込み層に通すのではなく、画像ごとに一回の畳み込み操作を行う。
    - 計算量を大幅に削減。(R-CNNの2-3.の処理の改善)
  - 物体候補領域の提案部分はSelective Searchを使用しているのでそこまで早くない。

## Faster R-CNN
- 関心領域の切り出し（物体候補領域の提案）もCNNで行う。
  - RPN（Region Proposal Network）
- ほぼリアルタイムで動作し、動画認識への応用も可能になった。
- 特徴マップ
  - 入力された画像はVGG16により特徴マップに変換される。
- 特徴マップにAnchor Points を仮視し，PointごとにAnchor Boxesを作成する
  - Anchor Points の個数：H × W
  - １つのAnchor Point あたりのAnchor Boxの個数：S
  - Anchor Boxes の個数：H × W × S
- RPNの出力
  - 各Anchor Boxes をGrand Truth のBoxesと比較し，含まれているものが背景か物体か，どれくらいズレてるか出力
  - 各Anchor Boxesにおいて背景か物体か：H × W ×S × 2
  - 各Anchor Boxesにおいて正解Boxesとのズレ（中心座標(x, y)，縦，横）：H × W × S × 4


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