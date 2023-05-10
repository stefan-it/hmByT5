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

| Language | Dataset                                                                                          | Additional Dataset                                                               |
|----------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| English  | [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)       | -                                                                                |
| German   | [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)       | -                                                                                |
| French   | [AjMC](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)       | [ICDAR-Europeana](https://github.com/stefan-it/historic-domain-adaptation-icdar) |
| Finnish  | [NewsEye](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-newseye.md) | -                                                                                |
| Swedish  | [NewsEye](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-newseye.md) | -                                                                                |
| Dutch    | [ICDAR-Europeana](https://github.com/stefan-it/historic-domain-adaptation-icdar)                 | -                                                                                |

## Overall Results

The following table shows performance (averaged F1-score on development set, 5 runs) for all models:

| Model                                                                                                                                                         | English AjMC | German AjMC  | French AjMC  | Finnish NewsEye | Swedish NewsEye | Dutch ICDAR  | French ICDAR | Avg. |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|--------------|--------------|-----------------|-----------------|--------------|--------------|------|
| [`stefan-it/byt5-small-english`](https://huggingface.co/stefan-it/byt5-small-english)                                                                         | 85.65 ± 1.21 | 87.27 ± 0.50 | 84.44 ± 0.79 |                 |                 |              |              |      |
| [`stefan-it/byt5-small-english-german`](https://huggingface.co/stefan-it/byt5-small-english-german)                                                           | 85.74 ± 0.72 | 87.45 ± 0.67 | 84.23 ± 0.65 |                 |                 |              |              |      |
| [`stefan-it/byt5-small-english-german-french`](https://huggingface.co/stefan-it/byt5-small-english-german-french)                                             | 85.61 ± 0.96 | 87.24 ± 0.76 | 84.39 ± 0.68 |                 |                 |              |              |      |
| [`stefan-it/byt5-small-english-german-french-finnish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish)                             | 85.30 ± 1.14 | 87.37 ± 0.53 | 84.12 ± 0.42 |                 |                 |              |              |      |
| [`stefan-it/byt5-small-english-german-french-finnish-swedish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish)             | 85.40 ± 0.78 | 87.12 ± 0.19 | 84.41 ± 0.34 |                 |                 |              |              |      |
| [`stefan-it/byt5-small-english-german-french-finnish-swedish-dutch`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish-dutch) | 85.51 ± 0.68 | 87.58 ± 0.39 | 84.39 ± 0.83 | 55.46 ± 1.99    | 73.38 ± 2.45    | 84.80 ± 0.44 | 75.97 ± 0.55 |      |
| [`stefan-it/byt5-small-multilingual-4g`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g)                                                         | 83.49 ± 0.96 | 87.65 ± 0.63 | 84.16 ± 0.90 |                 |                 |              |              |      |
| [`stefan-it/byt5-small-multilingual-4g-2e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-2e)                                                   | 83.86 ± 0.61 | 87.54 ± 0.19 | 84.29 ± 0.41 |                 |                 |              |              |      |
| [`stefan-it/byt5-small-multilingual-4g-3e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-3e)                                                   | 83.49 ± 0.99 | 87.38 ± 0.53 | 84.30 ± 0.51 |                 |                 |              |              |      |
| [`stefan-it/byt5-small-historic-multilingual-flax`](https://huggingface.co/stefan-it/byt5-small-historic-multilingual-flax)                                   | 83.28 ± 1.67 | 86.98 ± 0.71 | 83.49 ± 1.06 | 76.96 ± 1.58    | 78.80 ± 1.89    | 86.47 ± 0.79 | 77.43 ± 0.51 |      |

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

## Model: [`stefan-it/byt5-small-historic-multilingual-flax`](https://huggingface.co/stefan-it/byt5-small-historic-multilingual-flax)

Results with JAX/FLAX implementation on the multilingual model for 560k steps (0.5 epochs):

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   85.04 |   80.71 |   82.57 |   82.92 |   85.17 | 83.28 ± 1.67 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   83.7  |   82.12 |   83.12 |   84.16 |   83.1  | 83.24 ± 0.69 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   83.83 |   83.76 |   82.64 |   84.1  |   81.49 | 83.16 ± 0.97 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   78.65 |   83.69 |   80.97 |   84.02 |   82.21 | 81.91 ± 1.96 |

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

| Configuration                            | Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|-------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 87.72 |   87.61 |   87.56 |   87.66 |   87.17 | 87.54 ± 0.19 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 86.8  |   86.33 |   87.29 |   88.01 |   86.47 | 86.98 ± 0.61 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 85.44 |   86.95 |   85.92 |   85.51 |   86.27 | 86.02 ± 0.55 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 85.24 |   86.55 |   85.65 |   86.67 |   85.65 | 85.95 ± 0.56 |

## Model: [`stefan-it/byt5-small-multilingual-4g-3e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-3e)

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 87.92 | 87.48 | 87.93 | 86.95 | 86.6  | 87.38 ± 0.53 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 87.15 | 86.81 | 87.25 | 87.74 | 87.02 | 87.19 ± 0.31 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 85.89 | 86.67 | 85.58 | 86.53 | 85.61 | 86.06 ± 0.46 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 86.02 | 85.61 | 85.99 | 86.16 | 85.75 | 85.91 ± 0.2  |

## Model: [`stefan-it/byt5-small-historic-multilingual-flax`](https://huggingface.co/stefan-it/byt5-small-historic-multilingual-flax)

Results with JAX/FLAX implementation on the multilingual model for 560k steps (0.5 epochs):

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   87.3  |   88.16 |   86.63 |   86.74 |   86.06 | 86.98 ± 0.71 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   85.92 |   86.78 |   85.88 |   87.22 |   86.4  | 86.44 ± 0.51 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   85.31 |   85.04 |   84.83 |   85.68 |   84.9  | 85.15 ± 0.31 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   85.71 |   84.86 |   80.23 |   85.89 |   85.68 | 84.47 ± 2.15 |

## AjMC French

## Model: [`stefan-it/byt5-small-english`](https://huggingface.co/stefan-it/byt5-small-english)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   85.39 |   84.58 |   85.11 |   83.21 |   83.92 | 84.44 ± 0.79 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   84.54 |   83.71 |   83.19 |   83.67 |   83.52 | 83.73 ± 0.45 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   83.15 |   83.02 |   83.27 |   83.58 |   82.84 | 83.17 ± 0.25 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   82.88 |   82.9  |   82.63 |   83.59 |   83.38 | 83.08 ± 0.35 |

## Model: [`stefan-it/byt5-small-english-german`](https://huggingface.co/stefan-it/byt5-small-english-german)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   84.92 |   84.22 |   84.94 |   83.84 |   83.23 | 84.23 ± 0.65 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   84.01 |   82.77 |   83.6  |   84.04 |   83.99 | 83.68 ± 0.48 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   84.42 |   82.89 |   82.9  |   83.48 |   82.34 | 83.21 ± 0.71 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   82.51 |   83.65 |   81.94 |   83.23 |   84.25 | 83.12 ± 0.82 |

## Model: [`stefan-it/byt5-small-english-german-french`](https://huggingface.co/stefan-it/byt5-small-english-german-french)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   84.33 |   84.82 |   84.85 |   83.08 |   84.86 | 84.39 ± 0.68 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   84.62 |   84.42 |   84.01 |   83.67 |   83.98 | 84.14 ± 0.34 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   84.5  |   83.46 |   82.85 |   81.35 |   83.35 | 83.1 ± 1.03  |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   83.69 |   83.23 |   82.61 |   82.69 |   82.99 | 83.04 ± 0.39 |

## Model: [`stefan-it/byt5-small-english-german-french-finnish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   84.2  |   84.58 |   84.55 |   83.56 |   83.69 | 84.12 ± 0.42 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   84.16 |   84.01 |   83.81 |   83.21 |   83.56 | 83.75 ± 0.34 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   81.94 |   82.52 |   84.3  |   84.32 |   84.03 | 83.42 ± 1.0  |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   83.1  |   83.5  |   83.71 |   81.74 |   83.35 | 83.08 ± 0.7  |

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   84.42 |   84.92 |   84.57 |   84.21 |   83.91 | 84.41 ± 0.34 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   85.07 |   84.22 |   83.62 |   83.91 |   83.13 | 83.99 ± 0.65 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   81.7  |   84.28 |   82.94 |   83.71 |   83.58 | 83.24 ± 0.88 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   82.29 |   83.56 |   83.25 |   83.44 |   83.07 | 83.12 ± 0.45 |


## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish-dutch`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish-dutch)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   85.53 |   83.83 |   85.18 |   83.31 |   84.11 | 84.39 ± 0.83 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   83.31 |   84.71 |   84.07 |   84.03 |   83.05 | 83.83 ± 0.59 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   84.28 |   84.07 |   83.25 |   82.29 |   82.85 | 83.35 ± 0.74 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   82.72 |   83.95 |   81.59 |   84.2  |   82.16 | 82.92 ± 1.01 |

## Model: [`stefan-it/byt5-small-multilingual-4g`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   85.25 |   83.98 |   84.63 |   82.56 |   84.39 | 84.16 ± 0.9  |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   83.63 |   83.56 |   82.4  |   82.57 |   84.44 | 83.32 ± 0.75 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   82.37 |   83.12 |   81.35 |   83.33 |   81.69 | 82.37 ± 0.77 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   81.94 |   82.5  |   81.47 |   82.88 |   81.29 | 82.02 ± 0.6  |

## Model: [`stefan-it/byt5-small-multilingual-4g-2e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-2e)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   84.63 |   84.04 |   84.75 |   84.42 |   83.62 | 84.29 ± 0.41 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   83.17 |   82.04 |   83.6  |   84.09 |   83.12 | 83.2 ± 0.68  |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   82.24 |   83.29 |   81.74 |   82.62 |   82.34 | 82.45 ± 0.51 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   80.75 |   82.26 |   82.25 |   81.75 |   81.89 | 81.78 ± 0.55 |

## Model: [`stefan-it/byt5-small-multilingual-4g-3e`](https://huggingface.co/stefan-it/byt5-small-multilingual-4g-3e)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   84.11 |   83.98 |   85.32 |   83.98 |   84.11 | 84.3 ± 0.51  |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   84.47 |   83.15 |   84.17 |   83.73 |   82.85 | 83.67 ± 0.61 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   83.62 |   81.91 |   81.48 |   82.84 |   82.12 | 82.39 ± 0.75 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   83.04 |   81.24 |   81.64 |   82.43 |   82.19 | 82.11 ± 0.62 |

## Model: [`stefan-it/byt5-small-historic-multilingual-flax`](https://huggingface.co/stefan-it/byt5-small-historic-multilingual-flax)

Results with JAX/FLAX implementation on the multilingual model for 560k steps (0.5 epochs):

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   84.11 |   83.08 |   84.96 |   83.48 |   81.8  | 83.49 ± 1.06 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   83.31 |   81.64 |   82.12 |   82.91 |   83.42 | 82.68 ± 0.69 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   83.46 |   81.55 |   81.84 |   81.67 |   82.53 | 82.21 ± 0.71 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   81.44 |   82.53 |   80.64 |   83.29 |   82.27 | 82.03 ± 0.91 |

## NewsEye Finnish

## Model: [`stefan-it/byt5-small-historic-multilingual-flax`](https://huggingface.co/stefan-it/byt5-small-historic-multilingual-flax)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   77.73 |   74.68 |   79.05 |   77.71 |   75.63 | 76.96 ± 1.58 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   78.19 |   76.76 |   75.85 |   77.61 |   74.95 | 76.67 ± 1.17 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   74.47 |   66.95 |   71.34 |   53.76 |   65.81 | 66.47 ± 7.07 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   46.39 |   64.97 |   66.53 |   60.81 |   71.22 | 61.98 ± 8.48 |

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish-dutch`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish-dutch)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   57.61 |   55.51 |   52.95 |   57.72 |   53.53 | 55.46 ± 1.99 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   55.95 |   54.99 |   46.34 |   53.25 |   56.73 | 53.45 ± 3.74 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   46.06 |   40.57 |   46.37 |   45.74 |   48.05 | 45.36 ± 2.52 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   44.68 |   42.19 |   42.15 |   41.84 |   44.21 | 43.01 ± 1.18 |

## NewsEye Swedish

## Model: [`stefan-it/byt5-small-historic-multilingual-flax`](https://huggingface.co/stefan-it/byt5-small-historic-multilingual-flax)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   78.82 |   78.12 |   78.18 |   76.59 |   82.29 | 78.8 ± 1.89  |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   79.2  |   78.91 |   78.74 |   77.48 |   73.7  | 77.61 ± 2.04 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   74.46 |   64.86 |   71.25 |   63.91 |   65.23 | 67.94 ± 4.16 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   61.07 |   62.72 |   67.03 |   63.57 |   68.47 | 64.57 ± 2.75 |

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish-dutch`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish-dutch)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   73.19 |   70    |   75.09 |   76.95 |   71.68 | 73.38 ± 2.45 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   74.5  |   69.98 |   73.68 |   68.94 |   75.14 | 72.45 ± 2.5  |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   61.96 |   61.34 |   60.54 |   59.71 |   56.58 | 60.03 ± 1.88 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   61.26 |   57.35 |   61.93 |   61.2  |   57.91 | 59.93 ± 1.9  |

## ICDAR Europeana Dutch

## Model: [`stefan-it/byt5-small-historic-multilingual-flax`](https://huggingface.co/stefan-it/byt5-small-historic-multilingual-flax)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   85.01 |   87.32 |   86.41 |   86.71 |   86.91 | 86.47 ± 0.79 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   85.87 |   85.79 |   85.96 |   85.34 |   87.29 | 86.05 ± 0.66 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   85.41 |   85.91 |   87.2  |   85.14 |   86.46 | 86.02 ± 0.74 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   86.88 |   86.94 |   85.77 |   85.48 |   84.73 | 85.96 ± 0.85 |

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish-dutch`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish-dutch)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   84.59 |   84.89 |   85.58 |   84.7  |   84.25 | 84.8 ± 0.44  |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   84.41 |   84.63 |   83.83 |   84.89 |   84.75 | 84.5 ± 0.37  |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   84.14 |   82.28 |   85.45 |   85.87 |   84.2  | 84.39 ± 1.25 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   83.77 |   82.37 |   84.31 |   85.34 |   83.18 | 83.79 ± 1.01 |

## ICDAR Europeana French

## Model: [`stefan-it/byt5-small-historic-multilingual-flax`](https://huggingface.co/stefan-it/byt5-small-historic-multilingual-flax)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   77.6  |   77.77 |   76.48 |   77.95 |   77.34 | 77.43 ± 0.51 |
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   75.74 |   77.68 |   77.7  |   78.59 |   76.81 | 77.3 ± 0.96  |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   77.55 |   77.32 |   76.77 |   76.67 |   76.68 | 77.0 ± 0.37  |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   77.29 |   76.78 |   76.85 |   77.71 |   76.27 | 76.98 ± 0.49 |

## Model: [`stefan-it/byt5-small-english-german-french-finnish-swedish-dutch`](https://huggingface.co/stefan-it/byt5-small-english-german-french-finnish-swedish-dutch)

| Configuration                            |   Run 1 |   Run 2 |   Run 3 |   Run 4 |   Run 5 | Avg.         |
|------------------------------------------|---------|---------|---------|---------|---------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` |   75.34 |   76.69 |   75.84 |   75.44 |   76.52 | 75.97 ± 0.55 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` |   76.38 |   75.37 |   74.72 |   75.54 |   75.15 | 75.43 ± 0.55 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` |   74.35 |   76.02 |   75.92 |   75.33 |   74.85 | 75.29 ± 0.63 |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` |   74.82 |   74.93 |   74.89 |   75.82 |   74.02 | 74.9 ± 0.57  |

</details>

# Mean Noise Span Length

The previously pretrained hmByT5 models "accidentally" use a mean noise span length of 3, because this value is the
default one for T5. But the ByT5 paper mentions, that using a length of 3 would make pretraining tasks too easy, and
recommend a value of 20. We pretrained an English model with `mean_noise_span_length=20` and fine-tuned it on English
AjMC dataset:

| Configuration                            | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Avg.         |
|------------------------------------------|-------|-------|-------|-------|-------|--------------|
| `wsFalse-bs4-e10-lr0.00015-poolingfirst` | 85.48 | 84.6  | 85.65 | 86.83 | 86.53 | 85.82 ± 0.79 |
| `wsFalse-bs4-e10-lr0.00016-poolingfirst` | 85.35 | 84.5  | 86.05 | 85.1  | 85.18 | 85.24 ± 0.5  |
| `wsFalse-bs8-e10-lr0.00016-poolingfirst` | 84.14 | 83.45 | 84.4  | 84.9  | 85.82 | 84.54 ± 0.79 |
| `wsFalse-bs8-e10-lr0.00015-poolingfirst` | 85.27 | 85.3  | 83.33 | 85.25 | 81.7  | 84.17 ± 1.45 |

For comparison the model using a length of 3 achieved 85.65 ± 1.21. So we can also see performance improvements when
using `mean_noise_span_length=20`.

We also pretrained a monolingual model for Dutch on the Delpher corpus with both `mean_noise_span_length=3` and
`mean_noise_span_length=20` and will run performance comparisons soon.