# NER Fine-Tuning

We use Flair for fine-tuning NER models on the
[AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md) dataset from
[HIPE-2022 Shared Task](https://hipe-eval.github.io/HIPE-2022/).

All models are fine-tuned on A10 (24GB) instances from [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud) using
Flair:

```bash
$ git clone https://github.com/flairNLP/flair.git
$ cd flair
$ git checkout support_byt5
$ pip3 install -e .
$ cd
```

Clone this repo for fine-tuning NER models:

```bash
$ git clone https://github.com/stefan-it/hmByT5.git
$ cd hmByT5/bench
```

We use a config-driven hyper-parameter search. The script [`flair-fine-tuner.py`](flair-fine-tuner.py) can be used to
fine-tune NER models from our Model Zoo.

All configurations can be found under the `configs/ajmc` folder in this repository.

Example command for hyper-parameter search for the hmByT5 model on English part of AjMC corpus:

```bash
$ python3 flair-fine-tuner.py ./configs/ajmc/hmbyt5-small-en.json
```

# Preliminary Results

We evaluated the hmByT5 that was pretrained on English corpus for 200k steps:

| Hyper-param Configuration                | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.  |
|------------------------------------------|-------|-------|-------|-------|-------|-------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 83.80 | 84.78 | 83.74 | 83.35 | 84.37 | 84.01 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 84.67 | 82.69 | 83.92 | 84.53 | 82.90 | 83.74 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 82.12 | 83.82 | 83.37 | 83.00 | 83.70 | 83.20 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 83.45 | 82.83 | 84.15 | 81.76 | 83.78 | 83.19 |

It turns out, that the results are not on-par with current SOTA on the English AjMC corpus, see a comparison
[here](https://github.com/stefan-it/blbooks-lms#model-zoo). Thus, we continue experiments with the Hugging Face
Transformers JAX/FLAX implementation to pretrain ByT5 models on TPU.
