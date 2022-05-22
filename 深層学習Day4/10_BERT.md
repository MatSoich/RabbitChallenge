<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# BERT
- Bidirectional Transformerをユニットにフルモデルで構成したモデル
- 事前学習タスクとして、マスク単語予測タスク、隣接文判定タスクを与える
- BERTからTransfer Learningを行った結果、8つのタスクでSOTA達成
- Googleが事前学習済みモデルを公開済み

1. Background
   - 様々な自然言語処理タスクにおいて事前学習が有効
     - Featured based ApproachとFine Tuningの内、Fine TuningがBERTでは利用される。 
2. What is BERT?
   - Fine-tuningアプローチの事前学習に工夫を加えた
   - 双方向Transformer
     - tensorを入力としtensorを出力
     - モデルの中に未来情報のリークを防ぐためのマスクが存在しない
       - →従来のような言語モデル型の目的関数は採用できない(カンニングになるため)
       - →事前学習タスクにおいて工夫する必要がある!
   - Self-Attentionにより文脈を考慮して各単語をエンコード
     - 力を全て同じにして学習的に注意箇所を決めていく
   - 入力
     - 3種類のEmbeddingのSumを入力とする。（文章ごとにこの構造をとる）
       - トークン埋め込み：WordPieceでTokenizationしたものをEmbedding
       - 単語位置埋め込み：系列長1～512の表現
       - 文区別埋め込み：一文目、二分目の区別
     - 文章のペア or ひとつの文章の単位で入力は構成される。 (事前学習の際は必ずペア) 



3. Application
