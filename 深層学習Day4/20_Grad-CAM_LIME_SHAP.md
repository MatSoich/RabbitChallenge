<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# モデルの解釈性への挑戦
- なぜ解釈性が重要なのか
  - ディープラーニング活用の難しいことの１つは「ブラックボックス性」
  - 判断の根拠を説明できない
  - 実社会に実装する後に不安が発生「なぜ予測が当たっているのか」を説明できない
    - 例:「AIによる医療診断の結果、腫瘍は悪性ですが、AIがそう判断した根拠は解明できません」
  - モデルの解釈性に注目し、「ブラックボックス性」の解消を目指した研究が進められている
  - CAM,Grad CAM, LIME, SHAPの４つについて説明していく。
1. CAM
   - ネットワークの大部分が畳み込み層で構成されているネットワークかつ最終的な出力の前にGlobal Average Poolingを実施していれば実装可能。
   - あ
2. Grad-CAM
3. LIME
4. SHAP
