<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 3_4_spoken_digit.ipynbを実践しつつspoken_digitデータセットを例にした再起ニューラルネットワークの構築方法を学ぶ。

`spoken_digit`データセットは、`mnist`の音声版。
データの特徴は、


*   **長さがバラバラ**の2500個の音声ファイルが入っています。
*   それぞれの音声ファイルは、0から9の数字を読み上げたものです。
*   5人の話者が、各数字を50回ずつ読み上げています。
*   各音声ファイルには、対応する数字がラベルとして収録されています。

データセットの読込は、`tensorflow_datasets`を使って行います。`tensorflow_datasets`は予めある程度の前処理を行ったデータを手軽にダウンロード･準備･使用することができるライブラリです。

ライブラリの機能で、ダウンロードしたデータセットを、シャッフルすることや、分割することができます。

ここでは、データセットの全体の、


*   70%をトレーニング用
*   15%を検証用
*   15%をテスト用

として分割します。
