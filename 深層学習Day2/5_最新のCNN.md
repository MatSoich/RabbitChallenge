<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# AlexNetの例
- 最新ではないがAlex Netの例を紹介する。
![kakunin](imgs/AlexNet.png) 
- 入力には224 x 224の画像を使い、それを11x11のフィルターで畳み込み演算を行う。（多分ストライド４パディング３とか）
- 96(or32*RGB分？)種類のフィルター結果を55 x 55で表す。更に5x5のフィルターでMaxプーリングを行う。
- 96種類を基に256種類のプーリング結果を27x27で表す。更にそれを3x3のフィルターでマックスプーリングして384種類の13x13で表す。
- 更に全体の大きさを変えずに3x3で畳み込み、その後また、3x3畳み込みをして、256種類の13x13の結果をとする。
- その後横方向一列4096（13x13x256）の情報に並び替える。（Fratten）
  - その後のモデルでは、より効果の高いGlobal Average PoolingやGlobal Max Poolingが使われることの方が多い。（この例だと、13*13の情報の中で、最大の情報、あるいは平均の情報を256個並べるようなもの）
- そこから先は普通のニューラルネットワークのような数値の計算を行なっている。
- 全結合層の部分はドロップアウトを用いている。
