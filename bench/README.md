# NER Fine-Tuning

We use Flair for fine-tuning NER models on the
[AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md) dataset from
[HIPE-2022 Shared Task](https://hipe-eval.github.io/HIPE-2022/).

All models are fine-tuned on A10 (24GB) instances from [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud) using
Flair:

```bash
$ git clone -b support_byt5 https://github.com/flairNLP/flair.git
$ cd flair
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

To get a nice overview of the results (incl. best hyper-parameter configuration), just run the log parsing script:

```bash
$ python3 flair-log-parser.py "hipe2022-flert-fine-tune-ajmc-first-pooling/en-stefan-it/byt5-small-english-bs*"
```

# Preliminary Results

We evaluated the hmByT5 model that was pretrained on English corpus for 200k steps:

| Hyper-param Configuration                | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 83.80 | 84.78 | 83.74 | 83.35 | 84.37 | 84.01 ± 0.50 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 84.67 | 82.69 | 83.92 | 84.53 | 82.90 | 83.74 ± 0.82 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 82.12 | 83.82 | 83.37 | 83.00 | 83.70 | 83.20 ± 0.61 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 83.45 | 82.83 | 84.15 | 81.76 | 83.78 | 83.19 ± 0.84 |

It turns out, that the results are not on-par with current SOTA on the English AjMC corpus, see a comparison
[here](https://github.com/stefan-it/blbooks-lms#model-zoo). Thus, we continue experiments with the Hugging Face
Transformers JAX/FLAX implementation to pretrain ByT5 models on TPU.

Results with the Hugging Face Transformers JAX/FLAX implementation are really promising.
We evaluated a hmByT5 model that was pretrained on English corpus for one epoch:

| Hyper-param Configuration                | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 84.35 | 84.51 | 85.21 | 87.01 | 87.17 | 85.65 ± 1.21 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 85.78 | 85.03 | 86.40 | 85.48 | 84.47 | 85.43 ± 0.66 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 84.70 | 85.41 | 85.85 | 82.94 | 83.64 | 84.51 ± 1.08 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 83.23 | 85.95 | 85.41 | 83.02 | 84.16 | 84.35 ± 1.16 |

Results with JAX/FLAX implementation on the multilingual model (4GB of text per language) for one epoch:

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 84.66 | 84.10 | 81.79 | 83.45 | 83.47 | 83.49 ± 0.96 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 83.99 | 82.85 | 82.44 | 84.57 | 83.49 | 83.47 ± 0.76 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 81.96 | 82.05 | 82.52 | 82.13 | 83.08 | 82.35 ± 0.41 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 83.10 | 81.73 | 82.46 | 81.44 | 82.44 | 82.23 ± 0.59 |
