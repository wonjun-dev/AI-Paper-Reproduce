# SimCSE: Simple Contrastive Learning of Sentence Embeddings

This is a [SimCSE](https://arxiv.org/abs/2104.08821) implementation that performed sentence embedding using contrastive learning. This is an experiment that confirmed the effectiveness in the Relation Extraction task of the KLUE benchmark.

As a result of the experiment, the alignment and uniformity of the [CLS] token were improved, and faster convergence, f1 score, and auprc were improved in linear evaluation. 

## Results


<p float='left'>
<img src="./imgs/train_loss.png" width='400' height='250'/>
<img src="./imgs/eval_loss.png" width='400' height='250'/>
</p>

<p float='left'>
<img src="./imgs/eval_f1.png" width='400' height='250'/>
<img src="./imgs/eval_auprc.png" width='400' height='250'/>
</p>

## Discussion
Considering the class distribution, I observed that it works well when the mini-batches are equally configured.
I guess that it is because the class that occupies the majority is often regarded as a negative pair even though it is the same class.