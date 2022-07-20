<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

10 BERT
==========

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
- 環境設定

```python
from transformers import TFBertModel
from transformers import BertJapaneseTokenizer

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

bert = TFBertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

import MeCab
import numpy as np
import tensorflow as tf
import os
```

- データの準備

```python
with open('train.txt', 'r', encoding='utf-8') as f:
  text = f.read().replace('\n', '')
mecab = MeCab.Tagger("-Owakati")
text = mecab.parse(text).split()
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
seq_length = 128

# 訓練用サンプルとターゲットを作る
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(3):
    print(f'Input data: {repr("".join(idx2char[input_example.numpy()]))}')
    print(f'Target data: {repr("".join(idx2char[target_example.numpy()]))}')
```

- データのサンプルは以下。

> Input data: 'こころ夏目漱石-------------------------------------------------------【テキスト中に現れる記号について】《》：ルビ（例）私《わたくし》は｜：ルビの付く文字列の始まりを特定する記号（例）先生一人｜麦藁帽《むぎわらぼう》を［＃］：入力者注主に外字の説明や、傍点の位置の指定（数字は、JISX0213の面区点番号、または底本のページと行数）（例）※［＃「てへん'
> Target data: '夏目漱石-------------------------------------------------------【テキスト中に現れる記号について】《》：ルビ（例）私《わたくし》は｜：ルビの付く文字列の始まりを特定する記号（例）先生一人｜麦藁帽《むぎわらぼう》を［＃］：入力者注主に外字の説明や、傍点の位置の指定（数字は、JISX0213の面区点番号、または底本のページと行数）（例）※［＃「てへん＋'
> Input data: '劣」、第3水準1-84-77］-------------------------------------------------------［＃２字下げ］上先生と私［＃「上先生と私」は大見出し］［＃５字下げ］一［＃「一」は中見出し］私《わたくし》はその人を常に先生と呼んでいた。だからここでもただ先生と書くだけで本名は打ち明けない。これは世間を憚《はば》かる遠慮というよりも、その'
> Target data: '」、第3水準1-84-77］-------------------------------------------------------［＃２字下げ］上先生と私［＃「上先生と私」は大見出し］［＃５字下げ］一［＃「一」は中見出し］私《わたくし》はその人を常に先生と呼んでいた。だからここでもただ先生と書くだけで本名は打ち明けない。これは世間を憚《はば》かる遠慮というよりも、その方'
> Input data: 'が私にとって自然だからである。私はその人の記憶を呼び起すごとに、すぐ「先生」といいたくなる。筆を執《と》っても心持は同じ事である。よそよそしい頭文字《かしらもじ》などはとても使う気にならない。私が先生と知り合いになったのは鎌倉《かまくら》である。その時私はまだ若々しい書生であった。暑中休暇を利用して海水浴に行った友達からぜひ来いという端書《はがき》を受け取ったので、私は多少の金を工面《くめん》し'
> Target data: '私にとって自然だからである。私はその人の記憶を呼び起すごとに、すぐ「先生」といいたくなる。筆を執《と》っても心持は同じ事である。よそよそしい頭文字《かしらもじ》などはとても使う気にならない。私が先生と知り合いになったのは鎌倉《かまくら》である。その時私はまだ若々しい書生であった。暑中休暇を利用して海水浴に行った友達からぜひ来いという端書《はがき》を受け取ったので、私は多少の金を工面《くめん》して'

- 学習プロセス

```python
BATCH_SIZE = 64


BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

input_ids = tf.keras.layers.Input(shape=(None, ), dtype='int32', name='input_ids')
inputs = [input_ids]

bert.trainable = False
x = bert(inputs)

out = x[0]

Y = tf.keras.layers.Dense(len(vocab))(out)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

model = tf.keras.Model(inputs=inputs, outputs=Y)
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(loss=loss,
              optimizer=tf.keras.optimizers.Adam(1e-7))

model.fit(dataset,epochs=5, callbacks=[checkpoint_callback])
```

- 結果（今回は処理速度も問題もありEpochは五回のみ）

> Epoch 5/5
> 33/33 [==============================] - 24s 706ms/step - loss: 9.4642

- 文章生成

```python
def generate_text(model, start_string):
  # 評価ステップ（学習済みモデルを使ったテキスト生成）

  # 生成する文字数
  num_generate = 30

  # 開始文字列を数値に変換（ベクトル化）
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # 結果を保存する空文字列
  text_generated = []

  # 低い temperature　は、より予測しやすいテキストをもたらし
  # 高い temperature は、より意外なテキストをもたらす
  # 実験により最適な設定を見つけること
  temperature = 1

  # ここではバッチサイズ　== 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # バッチの次元を削除
      predictions = tf.squeeze(predictions, 0)

      # カテゴリー分布をつかってモデルから返された言葉を予測 
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # 過去の隠れ状態とともに予測された言葉をモデルへのつぎの入力として渡す
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (''.join(start_string) + ''.join(text_generated))

text = '私は'
mecab = MeCab.Tagger("-Owakati")
text = mecab.parse(text).split()
generate_text(model, text)

```

> '私は雪崩交際たし酒家昼間他愛えんとつ結ん封じできるみち無断取ん流れ響極る先ずあえて術語思い比べまち歩ん苦しむ弾ひる源単調かなり羽二重繋'

- 遠目で見ると漢字と平仮名のバランスは日本語に見えるものが出てきているが、学習が足りないので日本語にはなっていない。
