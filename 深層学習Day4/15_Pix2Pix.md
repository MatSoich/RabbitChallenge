<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# Pix2Pix
- 役割
  - Conditional GANと同様の考え方
  - 条件としてラベルではなく画像を用いる
  - 条件画像が入力され，何らかの変換を施した画像を出力する
  - 画像の変換方法を学習（以下のような組み合わせのペアを使い学習する。）
    - 着色していないものと着色しているもの
    - エッジ抽出したものと元画像
- Pix2Pixの工夫点
1. U-Net
   - Generatorに使用
   - 物体の位置を検出
   - セマンティックセグメンテーションにも使われている手法
2. L1正則化項の追加
   - Discriminatorの損失関数に追加
   - 普通のGANと異なり，pix2pixは画像の変換方法を学習するため条件画像と生成画像に視覚的一致性が見られる
   - 画像の高周波成分（色の変化が顕著な部分）を学習し，Generatorが生成した画像がぼやけることを防ぐ
3. PatchGAN
   - 条件画像をパッチに分けて，各パッチにPix2pixを適応
   - 正確な高周波成分の強調による視覚的一致性の向上
   - L１正則化項の効果を向上