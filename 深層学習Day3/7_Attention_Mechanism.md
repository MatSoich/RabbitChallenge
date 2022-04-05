<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# Attention Mechanism
- Seq2Seqの問題点
- 長い表現に対応できない。
  - 長い意味をベクトルに格納しきれない。
- Attention Mechanism
  - 文中の重要な情報を自力で取得する機構。
  - 近年評価が高い自然言語のモデルは全てAttention Mechanism

# 確認テスト２８
- RNNとword2vec、Seq2SeqとAttention Mechanismの違いを簡潔に述べよ。
  - RNNは時系列データを処理するのに適したニューラルネットワーク 
  - word2Vecは単語の分散表現ベクトルを得る手法
  - seq2seqは１つの時系列データから別の時系列データを得る手法。
  - Attentinon Mechanismは時系列データの中身のそれぞれの関連性について、重みをつける手法。

# 確認テスト２９
- Seq2SeqとHRED,HREDとVREDの違いについて簡潔に述べよ。
  - Seq2Seqは一問一答のようにある時系列データからある時系列データを作り出すネットワーク
  - HREDはSeqSeq２Seqに文脈の意味を足すことで、文脈の意味を汲み取ったENCODEとDECODEを可能にしている。
  - VHREはHREDが当たり障りのない文脈しか出さないことを修正するためにVAEの考えを取り入れて改良したもの。