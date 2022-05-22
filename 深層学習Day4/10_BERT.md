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
   - 事前学習
     - 空欄語予測と隣接文予測を同時に事前学習する
       - ->単語分散表現と文章分散表現を同時に獲得できる
     - 空欄語予測 (Masked Language Prediction)
       - 文章中の単語のうち15%がMASK対象に選ばれる 
       - => 選ばれた15%の単語の位置にフラグを立てる。
         - 選ばれた単語の内、80%が[MASK]に置き換えられ、10%が他の単語に置き換えられ、残り10%は置き換えない。
       - => 文章を入力としてフラグが付いている位置のオリジナルの入力単語が何であるかを出力する
         - (15%対象外の単語に関しては扱わない)事前学習 
     - 双方向学習でない理由
        - 双方向だと間接的に複数層の文脈から自分自身を見てしまう。
        - 一方でこの手法を採用すると単語全体の15%しか学習材料に使えないため学習に時間がかかってしま
   - 隣接文予測 (Next Sentence Prediction)
     - ２つの連なる文章ペアに対して、隣接文を50%の確率でシャッフルする
       - => 二つの文章を入力として、隣接文であるかのT/Fを出力する
     - データセット：BooksCorpus(800MB) + English Wikipedia(2500MB)
       - 入力文章の合計系列長が512以下になるように2つの文章をサンプリング 
       - Next Sentence Predictionのため、文章1と文章2の組み合わせは50%の確率で変わる 
       - MLMのためWordPieceトークンに分けられた後マスクされる

3. Application
- 8つのNLPベンチマークタスクについて、過去最高の成果。

# 実装
- 4-6_BERT.ipynbを実装。
- 