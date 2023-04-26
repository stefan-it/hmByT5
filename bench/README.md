# NER Fine-Tuning

We use Flair for fine-tuning NER models on the
[AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md) dataset from
[HIPE-2022 Shared Task](https://hipe-eval.github.io/HIPE-2022/).

All models are fine-tuned on A10 (24GB) instances from [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud) using
Flair:

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

All configurations can be found under the `configs/ajmc` folder in this repository.

Example command for hyper-parameter search for the hmByT5 model on English part of AjMC corpus:

```bash
$ python3 flair-fine-tuner.py ./configs/ajmc/en/hmbyt5-small-flax-en.json
```

To get a nice overview of the results (incl. best hyper-parameter configuration), just run the log parsing script:

```bash
$ python3 flair-log-parser.py "hipe2022-flert-fine-tune-ajmc-first-pooling/en-stefan-it/byt5-small-english-bs*"
```

# Small Benchmark

We test our pretrained language models on various datasets from HIPE-2020, HIPE-2022 and Europeana. The following table
shows an overview of used datasets.

| Language | Dataset                                                                                          | Additional Dataset                                                                                     |
|----------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| English  | [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)       | [Topres19th](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-topres19th.md) |
| German   | [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)       | [HIPE-2020](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-hipe2020.md)    |
| French   | [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)       | [HIPE-2020](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-hipe2020.md)    |
| Finnish  | [NewsEye](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-newseye.md) | -                                                                                                      |
| Swedish  | [NewsEye](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-newseye.md) | -                                                                                                      |
| Dutch    | [ICDAR-Europeana](https://github.com/stefan-it/historic-domain-adaptation-icdar)                 | -                                                                                                      |

## Overall Results

The following table shows performance (averaged F1-score on development set, 5 runs) for all models:

| Model                                                                                                                                                         | English AjMC | English Topres19th | German AjMC  | German HIPE-2020 | French AjMC | French HIPE-2020 | Finnish NewsEye | Swedish NewsEye | Dutch ICDAR | Avg. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|--------------------|--------------|------------------|-------------|------------------|-----------------|-----------------|-------------|------|
| [`stefan-it/byt5-small-english`](https://huggingface.co/stefan-it/byt5-small-english)                                                                         | 84.01 ± 0.50 |                    | 87.27 ± 0.50 |                  |             |                  |                 |                 |             |      |
| [`stefan-it/byt5-small-english-german`](https://huggingface.co/stefan-it/byt5-small-english-german)                                                           | 85.74 ± 0.72 |                    | 87.45 ± 0.67 |                  |             |                  |                 |                 |             |      |
| [`stefan-it/byt5-small-english-german-french`](https://huggingface.co/stefan-it/byt5-small-english-german-french)                                             | 85.61 ± 0.96 |                    | 87.24 ± 0.76 |                  |             |                  |                 |                 |             |      |
| [`stefan-it/byt5-small-english-german-french-finnish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish)                             | 85.30 ± 1.14 |                    | 87.37 ± 0.53 |                  |             |                  |                 |                 |             |      |
| [`stefan-it/byt5-small-english-german-french-finnish-swedish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish)             | 85.40 ± 0.78 |                    | 87.12 ± 0.19 |                  |             |                  |                 |                 |             |      |
| [`stefan-it/byt5-small-english-german-french-finnish-swedish-dutch`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish-dutch) | 85.51 ± 0.68 |                    | 87.58 ± 0.39 |                  |             |                  |                 |                 |             |      |
| [`stefan-it/byt5-small-multilingual-4g`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g)                                                         | 83.49 ± 0.96 |                    | 87.65 ± 0.63 |                  |             |                  |                 |                 |             |      |
| [`stefan-it/byt5-small-multilingual-4g-2e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-2e)                                                   | 83.86 ± 0.61 |                    |              |                  |             |                  |                 |                 |             |      |
| [`stefan-it/byt5-small-multilingual-4g-3e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-3e)                                                   | 83.49 ± 0.99 |                    | 87.38 ± 0.53 |                  |             |                  |                 |                 |             |      |

<details>
<summary>Detailed results</summary>

## AjMC English

## Model: [`stefan-it/byt5-small-historic-multilingual`](https://huggingface.co/stefan-it/byt5-small-historic-multilingual)

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

## Model: [`stefan-it/byt5-small-english`](https://huggingface.co/stefan-it/byt5-small-english)

Results with the Hugging Face Transformers JAX/FLAX implementation are really promising.
We evaluated a hmByT5 model that was pretrained on English corpus for one epoch:

| Hyper-param Configuration                | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 84.35 | 84.51 | 85.21 | 87.01 | 87.17 | 85.65 ± 1.21 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 85.78 | 85.03 | 86.40 | 85.48 | 84.47 | 85.43 ± 0.66 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 84.70 | 85.41 | 85.85 | 82.94 | 83.64 | 84.51 ± 1.08 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 83.23 | 85.95 | 85.41 | 83.02 | 84.16 | 84.35 ± 1.16 |

## Model: [`stefan-it/byt5-small-english-german`](https://huggingface.co/stefan-it/byt5-small-english-german)

We use the previous `stefan-it/byt5-small-english` model as initial checkpoint (incl. last learning rate and no
warm-up steps) and continue pretraining on the German corpus for one epoch:

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   85.37 |   85.75 |   86.7  |   86.26 |   84.62 | 85.74 ± 0.72 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   84.97 |   85.31 |   85.58 |   84.33 |   85.27 | 85.09 ± 0.43 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   84.64 |   84.62 |   85.04 |   83.92 |   85.24 | 84.69 ± 0.45 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   84.5  |   84.62 |   83.41 |   85.21 |   83.55 | 84.26 ± 0.68 |

## Model: [`stefan-it/byt5-small-english-german-french`](https://huggingface.co/stefan-it/byt5-small-english-german-french)

We use the previous English+German model as initial checkpoint (incl. last learning rate and no warm-up steps) and 
continue pretraining on the French corpus for one epoch:

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 86.19 | 86.8  | 84.58 | 86.12 | 84.36 | 85.61 ± 0.96 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 86.12 | 85.75 | 83.9  | 85.17 | 85.11 | 85.21 ± 0.75 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 85.37 | 85.24 | 84.9  | 84.56 | 84.81 | 84.98 ± 0.29 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 85.17 | 84    | 84.06 | 84.49 | 85.95 | 84.73 ± 0.74 |

## Model: [`stefan-it/byt5-small-english-german-french-finnish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish)

We use the previous English+German+French model as initial checkpoint (incl. last learning rate and no warm-up steps) and 
continue pretraining on the Finnish corpus for one epoch:

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 84.61 | 87.35 | 84.06 | 84.87 | 85.61 | 85.30 ± 1.14 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 86.4  | 84.51 | 83.86 | 84.66 | 85.48 | 84.98 ± 0.88 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 84.73 | 84.83 | 84.66 | 84.5  | 85.44 | 84.83 ± 0.32 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 84.66 | 85.37 | 85.2  | 82.96 | 84.91 | 84.62 ± 0.86 |

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish)

We use the previous English+German+French+Finnish model as initial checkpoint
(incl. last learning rate and no warm-up steps) and continue pretraining on the Swedish corpus for one epoch:

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 84.21 | 86.02 | 85.85 | 86.19 | 84.75 | 85.4 ± 0.78  |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 85.95 | 84.94 | 84.7  | 85.41 | 84.59 | 85.12 ± 0.5  |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 84.19 | 85.03 | 84.25 | 86.22 | 85.31 | 85.0 ± 0.75  |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 84.65 | 84.26 | 83.92 | 85.61 | 83.96 | 84.48 ± 0.62 |

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish-dutch`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish-dutch)

We use the previous English+German+French+Finnish+Swedish model as initial checkpoint
(incl. last learning rate and no warm-up steps) and continue pretraining on the Dutch corpus for one epoch:

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 84.35 | 85.34 | 85.71 | 86.46 | 85.68 | 85.51 ± 0.68 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 86.02 | 86.12 | 84.16 | 84.62 | 86.23 | 85.43 ± 0.86 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 84.36 | 84.93 | 84.8  | 84.46 | 85.55 | 84.82 ± 0.42 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 84.67 | 84.59 | 84.56 | 84.63 | 85    | 84.69 ± 0.16 |

## Model: [`stefan-it/byt5-small-multilingual-4g`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g)

Results with JAX/FLAX implementation on the multilingual model (4GB of text per language) for one epoch:

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 84.66 | 84.10 | 81.79 | 83.45 | 83.47 | 83.49 ± 0.96 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 83.99 | 82.85 | 82.44 | 84.57 | 83.49 | 83.47 ± 0.76 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 81.96 | 82.05 | 82.52 | 82.13 | 83.08 | 82.35 ± 0.41 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 83.10 | 81.73 | 82.46 | 81.44 | 82.44 | 82.23 ± 0.59 |

## Model: [`stefan-it/byt5-small-multilingual-4g-2e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-2e)

We use the previous 4GB model as initial checkpoint (incl. last learning rate and no warm-up steps) and 
continue pretraining on the same corpus for an additional epoch. Pretraining is currently running.

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 84.19 | 83.62 | 84.88 | 83.47 | 83.16 | 83.86 ± 0.61 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 83.57 | 83.59 | 82.37 | 85.58 | 81.73 | 83.37 ± 1.32 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 82.53 | 82.18 | 81.88 | 84.14 | 82.64 | 82.67 ± 0.78 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 82.44 | 79.4  | 83.04 | 81.91 | 82.49 | 81.86 ± 1.28 |

## Model: [`stefan-it/byt5-small-multilingual-4g-3e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-3e)

We use the previous 4GB model as initial checkpoint (incl. last learning rate and no warm-up steps) and 
continue pretraining on the same corpus for an additional epoch:

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 84.07 | 81.92 | 84.63 | 84.04 | 82.77 | 83.49 ± 0.99 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 84.27 | 84.06 | 84.71 | 82.11 | 80.85 | 83.2 ± 1.47  |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 82.16 | 81.3  | 81.73 | 84.63 | 81.75 | 82.31 ± 1.19 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 80.92 | 81.54 | 83.35 | 82.94 | 82.09 | 82.17 ± 0.89 |

## AjMC German

## Model: [`stefan-it/byt5-small-english`](https://huggingface.co/stefan-it/byt5-small-english)

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 86.91 | 87.26 | 86.57 | 87.98 | 87.62 | 87.27 ± 0.5  |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 87.08 | 86.47 | 86.02 | 86.77 | 87.43 | 86.75 ± 0.49 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 85.34 | 86.12 | 85.37 | 86.56 | 85.99 | 85.88 ± 0.47 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 85.58 | 85.04 | 85.44 | 84.36 | 85.92 | 85.27 ± 0.53 |

## Model: [`stefan-it/byt5-small-english-german`](https://huggingface.co/stefan-it/byt5-small-english-german)

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 87.29 | 88.01 | 87.17 | 86.43 | 88.33 | 87.45 ± 0.67 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 88.12 | 87.58 | 87.59 | 86.98 | 86.94 | 87.44 ± 0.44 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 86.78 | 86.63 | 85.85 | 86.64 | 85.82 | 86.34 ± 0.42 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 85.78 | 85.89 | 85.58 | 85.82 | 85.75 | 85.76 ± 0.1  |

## Model: [`stefan-it/byt5-small-english-german-french`](https://huggingface.co/stefan-it/byt5-small-english-german-french)

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 86.6  | 86.84 | 88.06 | 88.25 | 86.47 | 87.24 ± 0.76 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 86.13 | 86.54 | 87.98 | 86.67 | 87.53 | 86.97 ± 0.68 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 86.98 | 86.09 | 87.02 | 85.99 | 86.47 | 86.51 ± 0.43 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 86.75 | 86.33 | 84.53 | 85.78 | 85.85 | 85.85 ± 0.75 |


## Model: [`stefan-it/byt5-small-english-german-french-finnish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish)

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 87.71 | 86.98 | 87.08 | 86.85 | 88.25 | 87.37 ± 0.53 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 87.19 | 86.78 | 87.52 | 87.25 | 87.15 | 87.18 ± 0.24 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 86.19 | 86.23 | 86.47 | 87.15 | 86.91 | 86.59 ± 0.38 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 85.51 | 86.29 | 86.05 | 85.27 | 86.8  | 85.98 ± 0.55 |

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish)

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 87.12 | 86.88 | 87.43 | 87.17 | 86.98 | 87.12 ± 0.19 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 88.35 | 86.33 | 87.25 | 86.33 | 86.78 | 87.01 ± 0.75 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 86.74 | 86.84 | 86.67 | 86.98 | 86.26 | 86.7 ± 0.24  |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 85.85 | 85.47 | 86.54 | 85.85 | 84.1  | 85.56 ± 0.81 |

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish-dutch`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish-dutch)

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 87.66 | 88.19 | 87.56 | 86.95 | 87.53 | 87.58 ± 0.39 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 87.66 | 86.84 | 87.5  | 86.53 | 87.23 | 87.15 ± 0.42 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 86.33 | 87.19 | 87.85 | 86.9  | 86.81 | 87.02 ± 0.5  |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 86.34 | 84.71 | 85.75 | 85.89 | 85.85 | 85.71 ± 0.54 |

## Model: [`stefan-it/byt5-small-multilingual-4g`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g)

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 88    | 88.7  | 87.45 | 87.09 | 86.99 | 87.65 ± 0.63 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 87.68 | 87.74 | 87.02 | 87.72 | 87.14 | 87.46 ± 0.31 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 86.54 | 86.57 | 86.64 | 86.5  | 86.53 | 86.56 ± 0.05 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 86.06 | 86.23 | 85.44 | 86.63 | 86.67 | 86.21 ± 0.45 |

## Model: [`stefan-it/byt5-small-multilingual-4g-2e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-2e)

## Model: [`stefan-it/byt5-small-multilingual-4g-3e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-3e)

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 87.92 | 87.48 | 87.93 | 86.95 | 86.6  | 87.38 ± 0.53 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 87.15 | 86.81 | 87.25 | 87.74 | 87.02 | 87.19 ± 0.31 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 85.89 | 86.67 | 85.58 | 86.53 | 85.61 | 86.06 ± 0.46 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 86.02 | 85.61 | 85.99 | 86.16 | 85.75 | 85.91 ± 0.2  |


## AjMC French

## Model: [`stefan-it/byt5-small-english`](https://huggingface.co/stefan-it/byt5-small-english)

## Model: [`stefan-it/byt5-small-english-german`](https://huggingface.co/stefan-it/byt5-small-english-german)

## Model: [`stefan-it/byt5-small-english-german-french`](https://huggingface.co/stefan-it/byt5-small-english-german-french)

## Model: [`stefan-it/byt5-small-english-german-french-finnish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish)

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish)

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish-dutch`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish-dutch)

## Model: [`stefan-it/byt5-small-multilingual-4g`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g)

## Model: [`stefan-it/byt5-small-multilingual-4g-2e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-2e)

## Model: [`stefan-it/byt5-small-multilingual-4g-3e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-3e)

</details>
