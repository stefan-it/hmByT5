# NER Fine-Tuning

We use Flair for fine-tuning NER models on the
[AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md) dataset from
[HIPE-2022 Shared Task](https://hipe-eval.github.io/HIPE-2022/).

All models are fine-tuned on A10 (24GB) and A100 (40GB) instances from
[Lambda Cloud](https://lambdalabs.com/service/gpu-cloud) using Flair:

```bash
$ git clone -b support_byt5 https://github.com/flairNLP/flair.git && cd flair && pip3 install -e .
$ cd
```

Clone this repo for fine-tuning NER models:

```bash
$ git clone https://github.com/stefan-it/hmByT5.git
$ cd hmByT5/bench
```

We use a config-driven hyper-parameter search. The script [`flair-fine-tuner.py`](flair-fine-tuner.py) can be used to
fine-tune NER models from our Model Zoo.

Preliminary experiments can be found [here](README-PRELIMINARY.md).

# Small Benchmark

We test our pretrained language models on various datasets from HIPE-2020, HIPE-2022 and Europeana. The following table
shows an overview of used datasets.

| Language | Dataset                                                                                          | Additional Dataset                                                               |
|----------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| English  | [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)       | -                                                                                |
| German   | [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)       | -                                                                                |
| French   | [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)       | [ICDAR-Europeana](https://github.com/stefan-it/historic-domain-adaptation-icdar) |
| Finnish  | [NewsEye](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-newseye.md) | -                                                                                |
| Swedish  | [NewsEye](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-newseye.md) | -                                                                                |
| Dutch    | [ICDAR-Europeana](https://github.com/stefan-it/historic-domain-adaptation-icdar)                 | -                                                                                |
