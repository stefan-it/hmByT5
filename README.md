# hmByT5

Upcoming Historic Multilingual ByT5 Model. It covers the following languages:

* English (British Library Corpus - Books)
* German (Europeana Newspaper)
* French (Europeana Newspaper)
* Finnish (Europeana Newspaper)
* Swedish (Europeana Newspaper)
* Dutch (Delpher Corpus)

# Pretraining

We pretrain hmByT5 on a v3-32 TPU Pod. Details about the training can be found [here](hmbyt5/README.md). Additionally,
we perform pretraining using Transformers JAX/FLAX example, details can be found [here](hmbyt5-flax/README.md).

# Evaluation on Downstream Tasks (NER)

We use Flair to fine-tune hmByT5 on HIPE-2022 data. Details about the fine-tuning can be found [here](bench/README.md).

# **New**: Logbook

* 11.04.2022: Experiment with pretraining an initial English model with JAX/FLAX implementation from Transformers has
              finished. Results on AjMC are really promising and performance is ~1.65% better than with the original
              ByT5 implementation, and the model was only pretrained for one epoch over the English corpus! Results
              can be found [here](bench/README.md). The model is also available on the [Model Hub](https://huggingface.co/stefan-it/byt5-small-english).
              We are now using this model as inital checkpoint to continue pretraining with the German corpus.
* 09.04.2022: Preliminary experiments on English AjMC show, that the pretrained model on English corpus is not on-par
              with current SOTA. For that reason, we are trying the Hugging Face Transformers JAX/FLAX implementation
              to pretrain models. Details can be found in [this readme](hmbyt5-flax/README.md). Pretraining has already
              started.
* 07.04.2022: Pretraining for 200k steps on the English corpus finished without crashes! TensorBoard logs can be found
              on the [Model Hub](https://huggingface.co/stefan-it/byt5-small-historic-multilingual/tensorboard). We
              also uploaded all checkpoints (we checkpoint every 25k steps)
              [here](https://huggingface.co/stefan-it/byt5-small-historic-multilingual). Fine-Tuning experiments (NER)
              on the English part of AjMC corpus from HIPE-2022 are running.
* 03.04.2022: We start ByT5 pretraining with official ByT5 implementation on a v3-32 TPU Pod - thankfully provided by
              [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC) program. Plan is to pretrain on the
              English corpus for 200k steps and use the original ByT5 Small model as init checkpoint.

# Acknowledgements

Research supported with Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).
Many Thanks for providing access to the TPUs ❤️
