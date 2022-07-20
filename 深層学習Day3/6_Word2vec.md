<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

Word2vec
=========

# Word2Vec

- ニューラルネットワークを通して、単語の分散表現を獲得するための手法
  - 単語をone-hotベクトルで表現する
  - 次元数が多くメモリを大量に消費したり、似た単語の類似度を測ることができなかったりするなどの問題がある
  - より少ない次元数で単語の意味を数値で表現でき、現実的な計算量で省メモリ化を実現することができる。
  - RNN系のモデルでは通常、Word2Vecにより単語を分散表現に変換したものを入力として扱う。

# 実装

- LSTMのところで一度実装済みだが、簡易のものを再度実装する。
- 前処理

```terminal
curl http://public.shiroyagi.s3.amazonaws.com/latest-ja-word2vec-gensim-model.zip > latest-ja-word2vec-gensim-model.zip
unzip latest-ja-word2vec-gensim-model.zip
```

- gensim3.8.3を使用して実装

```python
from gensim.models.word2vec import Word2Vec
model_path = 'word2vec.gensim.model'
model = Word2Vec.load(model_path)
for r in model.wv.most_similar(positive=['野球']):
    print(r)
```

- 下記のように単語をone-hotベクトルで表現できた。

> ('ソフトボール', 0.8520911931991577)
> ('ラグビー', 0.8089540600776672)
> ('サッカー', 0.7957996726036072)
> ('軟式野球', 0.7862450480461121)
> ('少年野球', 0.7850692868232727)
> ('アメフト', 0.7838374972343445)
> ('プロ野球', 0.7779729962348938)
> ('リトルリーグ', 0.7625014781951904)
> ('ホッケー', 0.7582970261573792)
> ('フットサル', 0.7554671764373779)
