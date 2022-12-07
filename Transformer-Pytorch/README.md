# Attention is All You Need

This is a [Transformer](https://arxiv.org/abs/1706.03762) implementation that performed German-English(English-German) translation. I followed style of offical Pytorch implementation for learning their code structure of layers and modules.

## Results
|Metric|Train|Valid|
|---|---|---|
|CrossEntropy|1.067|1.600|
|Perplexity|2.907|4.957|

|Src|Pred|Target (Google Translate)|
|---|---|---|
|Zwei Männer spielen Fußball|Two men are playing soccer|Two men are playing soccer|
|Auf dem Rasen spielen zwei Männer Fußball|The two men are playing soccer on the grass|Two men are playing football on the lawn|
|Zwei Männer spielen an einem regnerischen Tag auf dem Rasen Fußball|Two men are playing soccer on a busy day|Two men play soccer on the grass on a rainy day|
|Deutschland ist schlechter im Fußball als Korea.|A soccer team is \<unk> a soccer ball while the kicker watches|Germany is worse in soccer than Korea|
|Die Frau schaut aus dem Fenster und der Mann liest ein Buch|The woman is looking out the window and the man is reading a book|The woman is looking out the window and the man is reading a book|