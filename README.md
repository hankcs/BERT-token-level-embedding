# BERT token level embedding

This tiny script demonstrates how to generate BERT token level embeddings. As BERT use word piece as tokenizer, OOV will be segmented into several word pieces. This script has implemented `2` strategies to handle that case.

In the following chapters, we'll show how to generate BERT embeddings for each token without pain.

## 1. Install BERT server and download pre-trained BERT model

Run following command:

```bash
pip3 install bert-serving-server  # server
pip3 install bert-serving-client  # client, independent of `bert-serving-server`
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
cd uncased_L-12_H-768_A-12/
bert-serving-start -model_dir . -pooling_strategy NONE -show_tokens_to_client -max_seq_len 256
```

## 2. Generate embeddings

Put the text you want to embed into `input.txt` :

```
Hello World !
I'm hankcs
```

Run `bert_token_embed.py`, you will get a pickle file called `output.pkl`. It stores a list of `numpy.ndarray`

```
<class 'list'>: [array([[-0.10005677,  0.10111555,  0.3707362 , ..., -0.79261583,
        -0.29630244, -0.43822828],
       [-0.1717627 , -0.08768683,  0.57421064, ..., -0.43223655,
        -0.02188881, -0.2638072 ],
       [-1.2841692 , -1.4125137 , -0.92776453, ...,  0.26873824,
        -0.03885475,  0.14489302]], dtype=float32), array([[-0.28254002, -0.01273985, -0.2916504 , ..., -0.99867177,
         0.7456796 ,  0.3703635 ],
       [-0.8595818 , -0.09063847, -0.14206652, ..., -0.10226044,
         0.4216262 , -0.20428266]], dtype=float32)]
```

The i-th `ndarray` is the BERT embedding of i-th sentence, of size n x c, where n is the sentence length, c is the dimension of BERT embedding(usually `768` or `1024`). For example, `Hello World !` is embedded as `3x768` tensor, since it contains `3` tokens.

As you may notice, `I'm hankcs` is embedded as `2x768` tensor. My ID `hankcs` is a typical OOV, which is segmented to `2` word pieces.

```
'hank', '##cs'
```

Each word piece has its own vector. There are `2` strategies implemented as `2` functions to merge them into one:

1. `embed_last_token`: Use the last word piece as the representation of whole token.
2. `embed_sum`: Use the average vector instead.

## Licence

#### Apache License 2.0

