<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 物体検知
- 最新手法だけ学んでも半年後、1年後、新しい手法に取って変わられることも多い。アルゴリズムの精度と評価を重点的に話す。
- 画像分類関連のタスク種類
  - 分類
    - 何の画像か
  - 物体検知
    - 画像内の物体・個体の識別
  - セマンティックセグメンテーション
    - 各ピクセルレベルの属性
  - インスタンスセグメンテーション
    - 各ピクセルレベルの属性＋属性内での個体の差異
- 画像分類関連タスクの性能を測るための指標
  - 代表的データセット
    - VOC12
      - PASCAL VOC Object Detection Challangeで使用されていたデータセット。主要貢献者が2012年に亡くなったため、コンペもそこで終了。
      - VOCはVisual Object Classesの意味
      - INSTANCE ANNOTATION（物体個々にラベリングが与えられている。）
      - クラス
        - 20
      - Train + Val
        - 11540
      - ???
        - 2.4
      - サイズ
        - 470 X 380
    - ILSVRC17
      - ILSVRC Object Detection Challangeで使用されていたデータセット
      - ILSVRCはImageNet Large Scale Visual Recognition Challangeの略
      - ImageNetのサブクラス
      - コンペは2017年に終了し、後継として、Open Images Challangeが誕生。（ある程度の精度が出てしまったため。）
      - クラス
        - 200
      - Train + Val
        - 476668
      - ???
        - 1.1
      - サイズ
        - 500 X 400

    - MS COCO18
      - MS COCO Object Detection Challangeで使用されたデータセット
      - 物体位置推定に対する新たな指標を提案している。
      - INSTANCE ANNOTATION（物体個々にラベリングが与えられている。）
      - クラス
        - 80
      - Train + Val
        - 123287
      - ???
        - 7.3
      - サイズ
        - 640 X 480
    - OICOD18
      - INSTANCE ANNOTATION（物体個々にラベリングが与えられている） 
      - Open Image Challange Object Detectionで使用されたデータセット
      - ILSVRCやMS COCOとは異なるアノテーションプロセス
      - Open Image V4（6000クラス以上/900万枚以上）のサブセット
      - クラス
        - 500
      - Train + Val
        - 1743042
      - ???
        - 7.0
      - サイズ
        -  バラバラ
- ???は画像あたりのBBoxの数（一枚当たりの物体数）
  - 数が大きいというのはいろんな物体が写真に入っているので、アイコン的な写りではなく、日常生活の風景に近いと考えられる。
- 用途に応じてBox /画像の選択をするのが望ましい。
- クラス数が多いことは良いことか？
  - ImageNetなどで無意味なクラス分けをしている場合もある。

# 分類問題における精度評価指標
混同行列の理解
 - True Positive
 - False Negative
 - False Positive
 - True Negative
 - Precision = TP / (TP + FP)
 - Recall = TP / (TP + FN)
 - Confidenceの閾値を変化させることでPrecision-Recall Curveが描ける
   - 閾値（threshold）を変化させたときのPR Curveを見て、どのように閾値を置くのが正しいいのか検討することができる。
 - 物体検出におけるconfidence matrixに関する注意点
   - 画像分類含むクラス分類では、閾値を変化させても、混同行列内の総数は変化しない。
   - 一方で物体検出では、混同行列内の総数が閾値を上げると減っていく。
     - 上記の点を理解して、閾値を設定する必要あり
 - IoU
   - 近藤行列によるラベルの精度とは別に物体検出においてはラベルの位置も重要
     - その指標として使われているのがIoU(Area of Overlap / Area of Union)
     - IoUを物体の位置に関する混同行列で表すと、IoU = TP/(TP+FP+FN)。Jaccord係数ともいう。
 - ラベルのコンフィデンスおよびIoUにおいてそれぞれ閾値を設定する必要がある。
 - AP
   - 定義は以下(意味はPR曲線の下側面積)
   > \\\(\displaystyle AP=\int_0^1 P(R)dR\\\)
 - mAP（ラベルごとに計算されるAPの平均がmAP）
   >\\\(\displaystyle mAP = \frac{1}{C} \sum_{i=1}^c {AP}_i\\\)
 - \\\(mAP_{COCO}\\\)
   - IoU閾値をここまでは0.5で固定していたものを0.5から0.95まで0.05刻みでAP&mAPを計算し、算術平均を撮ったもの
   > \\\(\displaystyle mAP_{COCO} = \frac{mAP_{0.5} + mAP_{0.55} + \cdots + mAP_{0.95} }{10}\\\)

# 検出速度の指標
- FPS(Flames Per Second)
- 使用するデータも重要

# 物体検出のフレームワーク
- 2段階検出器
  - 候補領域と検出とクラス推定を別々に行う。
  - 相対的に精度が高い
  - 相対的に計算量が多い推論も遅い傾向
- 1段階検出器
  - 候補領域と検出とクラス推定を同時に行う。
  - 相対的に精度が低い
  - 相対的に計算量が少なく推論も速い傾向

# SSD
- SSDは一段階検出器
- SSDの元ネットワークはVGG16
  - 16というのはConvolution層とFully Connected層の合計であり、max poolingやsoftmaxは含んでいないことに注意。
  - max poolingを通して、サイズやチャネル数を変化させていることがわかる。
- SSDのネットワーク構造のVGG16からの変更点
（画像）
- ネットワーク内の画像のサイズは、実際のサイズではなく、解像度を変えている点に注意。解像度を徐々に下げることで小さい物体の検知→大きい物体の検知を実現している。
- BBoxに関する工夫
  - Non Maximum Supression
  - Hard Negative Mining
- 損失関数について
- SSDの進化について

# Semantic Segmentationの概略
- Upsamplingを以下に実施していくかが重要なポイントだった。
  - なぜUpsamplingが重要か
    - 受容野を広げるため
      - 需要やを広げる手段
        - ①深いConvolution層
        - ②プーリング（ストライド）
          - ①は多層化に伴う演算量やメモリの問題が常に発生するため、②が採用されることも多い。
- Upsamplingの手法 
  - Decnvolution
  （画像）
  - U-Net
  （画像）
  - Unpooling
（画像）
- Upsampling以外で受容野を広げる方法
  - Dilated Convolution