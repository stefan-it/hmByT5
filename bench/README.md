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

Example command for hyper-parameter search for the hmByT5 model:

```bash
$ python3 flair-fine-tuner.py ./configs/hmbyt5-small-historic-multilinual.json
```
