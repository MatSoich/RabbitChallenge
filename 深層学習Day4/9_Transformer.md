<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# Seq2Seqの実践
- 4_3_lecture_chap1_exercise_public.ipynbで実装。
# Transformer
- ニューラル翻訳機構の弱点
  - 長さに対応できない
- 対応策として考えられたのがAttention
- Attention
  - CNNと似ているとよく言われる。
  - 局所的な位置しか参照できないCNNと違い、系列内の任意の点を参照できる。
- Source Target Attention と Self Attentionの違い
  - Source Target Attentionは受け取った情報に対して、狙った情報が近いものを見つける。
  - Self Attentionは自分で自分を学習することで最適な重みづけを見つける。
  - 周辺情報を考慮して重みが決められるので、CNNに考え方が少し近いかも。
- TransformerのEncoder部分
  - 6層
- TransfoemerのDecoder部分
  - 6層
  - Encoderの入力に対してはSource Target Attentionを使用する。過去のDecoder情報の入力に対してはSelf Attentionを使用する。
- Transformerの主要モジュール
  1. Positional Encoding
    - RNNを用いないので単語列の語順情報を追加する必要がある
      - 単語の位置情報をエンコード
        - \\\(\displaystyle PE_{(pos,2i)} = sin\left(\frac{pos}{10000^{\frac{2i}{512}}}\right)\\\)
        - \\\(\displaystyle PE_{(pos,2i+1)} = cos\left(\frac{pos}{10000^{\frac{2i}{512}}}\right)\\\)
      - posの(ソフトな)２進数表現
  2. Scaled Dot product Attention & Multi-Head Attention
     - Scaled Dot Product Attention
       - 全単語に関するAttentionをまとめて計算する
       - \\\(\displaystyle Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V\\\)
     - Multi-Head Attention
       - ８個のScaled Dot-Product Attentionの出力をConcat
       - それぞれのヘッドが異なる種類の情報を収集

  3. Positional Feed Forward Network
     - 位置情報を保持したまま順伝播させる
       - 各Attention層の出力を決定
         - ２層の全結合NN
         - 線形変換→ReLu→線形変換
     - \\\(FFN(x) = max(0,xW_1 + b_1 ) W_2 + b_2 \\\)
     - \\\(W_1 \in \R^{512X2048}, b_1 \in \R^{2048} \\\)
     - \\\(W_2 \in \R^{2048X512}, b_2 \in \R^{512} \\\)
  4. Masked Multi-Head Attention
     - 未来の情報を加えてはダメなので、一部の情報をマスクする。

# Transformerの実装
- 4_4_lecture_chap2_exercise_public.ipynbで実装。