
#### Results on UD1.3 (May 2017)

The table below provides results on UD1.3 (iters=20, h_layers=1, sgd) with
bilty ported to DyNet (2.0).

+poly is using pre-trained embeddings to initialize
word embeddings.  Note that for some languages it slightly hurts performance.

```
python src/bilty.py --dynet-seed 1512141834 --dynet-mem 1500 --train /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-train.conllu --test /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-test.conllu --dev /home/$user/corpora/pos/ud1.3/orgtok/goldpos//en-ud-dev.conllu --output /data/$user/experiments/bilty/predictions/bilty/en-ud-test.conllu.bilty-en-ud1.3-poly-i20-h1 --in_dim 64 --c_in_dim 100 --trainer sgd --iters 20 --sigma 0.2 --save /data/$user/experiments/bilty/models/bilty/bilty-en-ud1.3-poly-i20-h1.model --embeds embeds/poly_a/en.polyglot.txt --h_layers 1 --pred_layer 1  > /data/$user/experiments/bilty/nohup/bilty-en-ud1.3-poly-i20-h1.out 2> /data/$user/experiments/bilty/nohup/bilty.bilty-en-ud1.3-poly-i20-h1.out2
```

| Lang | i20-h1  | +poly |
| ---| -----:| -----:|
| ar | 96.15 | 96.43 |
| bg | 98.10 | 97.85 |
| ca | 98.13 | 98.19 |
| cs | 98.46 | 98.52 |
| cu | 96.55 | -- |
| da | 96.01 | 96.06 |
| de | 93.02 | 93.59 |
| el | 98.01 | 98.13 |
| en | 94.43 | 95.00 |
| es | 95.08 | 95.46 |
| et | 95.80 | 96.35 |
| eu | 94.69 | 95.16 |
| fa | 96.90 | 97.49 |
| fi | 94.60 | 95.46 |
| fr | 96.01 | 96.44 |
| ga | 90.40 | 91.23 |
| gl | 97.10 | -- |
| got | 95.60 | -- |
| grc | 93.91 | -- |
| he | 95.59 | 96.92 |
| hi | 96.48 | 96.95 |
| hr | 95.05 | 96.15 |
| hu | 94.19 | -- |
| id | 93.17 | 93.34 |
| it | 97.53 | 97.78 |
| kk | 74.45 | -- |
| la | 92.51 | -- |
| lv | 90.39 | -- |
| nl | 89.70 | 90.31 |
| no | 97.45 | 97.90 |
| pl | 96.03 | 97.22 |
| pt | 97.08 | 97.38 |
| ro | 95.86 | -- |
| ru | 95.51 | -- |
| sl | 97.50 | 96.41 |
| sv | 96.43 | 96.32 |
| ta | 85.82 | -- |
| tr | 94.25 | -- |
| zh | 92.87 | -- |

Using pre-trained embeddings often helps to improve accuracy, however, does not
strictly hold for all languages.

Using 'adam' instead of 'sgd' might help for cases where little
training data is available (like 'kk').

For more information, predictions files and pre-trained models
visit [http://www.let.rug.nl/bplank/bilty/](http://www.let.rug.nl/bplank/bilty/)



