# hmByT5

[![🤗](logo.jpeg "🤗")](https://github.com/stefan-it/hmByT5)

Upcoming Historical Multilingual and Monolingual ByT5 Models. Following languages will be covered:

* English (British Library Corpus - Books)
* German (Europeana Newspaper)
* French (Europeana Newspaper)
* Finnish (Europeana Newspaper)
* Swedish (Europeana Newspaper)
* Dutch (Delpher Corpus)
* Norwegian (NCC)

# Changelog

* 24.05.2023: We created two new organisations at Hugging Face Model Hub: [hmByT5 Preliminary](https://huggingface.co/hmbyt5-preliminary)
              and [hmByT5](https://huggingface.co/hmbyt5), where preliminary models and final models are uploaded.

# Pretraining

We pretrain hmByT5 on v3-8 and v3-32 TPUs. Details about the training can be found [here](hmbyt5/README.md).
Additionally, we perform pretraining using Transformers JAX/FLAX example, details can be found
[here](hmbyt5-flax/README.md).

# Preliminary Experiments

## Evaluation on Downstream Tasks (NER)

We use Flair to fine-tune hmByT5 on historic NER dataset. Details about the fine-tuning can be found
[here](bench/README-PRELIMINARY.md).

## **New**: Logbook

* 21.05.2023: Pretraining of ByT5 Base models for English and Dutch are completed. We already performed fine-tuning
              experiments for English, Dutch will follow. The preliminary experiments with the pretraining corpus of
              six languages are now completed. We are building a new and documented pretraining corpus now, that also
              includes Norwegian. Experiments will follow soon.
* 14.05.2023: Add results for two trained monolingual models for Dutch. It turns out that using a `mean_noise_span_length=20`
              slightly underperforms a model pretrained with `mean_noise_span_length=3` on Dutch par of ICDAR Europeana.
* 10.05.2023: Finalizing datasets and languages for small benchmark. We also performed experiments using the proposed
              `mean_noise_span_length=20` from the ByT5 paper. On English AjMC dataset we can see a performance boost
              when using a length of 20.
* 29.04.2023: Training of first [Multilingual model](https://huggingface.co/stefan-it/byt5-small-historic-multilingual-flax)
              crashed after 11 days with 560k steps trained. We will perform evaluations with this model now.
* 25.04.2023: Experiments with continued pretraining are completed. It turns out, that the strategy worked for
              English+German+French. However, after using Finnish the eval results for all following languages (Finnish,
              Swedish and Dutch) to really worse. Training loss is also very high. So this strategy seems to be not
              working with the used hyper-parameters (adjusted learning rate). Additionally, on 18th of April we started
              to pretrain a multilingual model on the complete 165GB training corpus. Model is available
              [here](https://huggingface.co/stefan-it/byt5-small-historic-multilingual-flax) and still training.
* 14.04.2023: Experiment with the English+German model has completed. Experiments show, that continuing the pretraining
              on another corpus (German) has no negative performance impact on English downstream task! On the contrary,
              result on English AjMC corpus led to slightly better results. We continue this experiment and pretrain
              on French corpus. Additionally, we pretrain the mulitlingual model (reduced 4GB of text for all languages)
              for an additional epoch. Result can be seen [here](bench/README-PRELIMINARY.md).
* 11.04.2023: Experiment with pretraining an initial English model with JAX/FLAX implementation from Transformers has
              finished. Results on AjMC are really promising and performance is ~1.65% better than with the original
              ByT5 implementation, and the model was only pretrained for one epoch over the English corpus! Results
              can be found [here](bench/README-PRELIMINARY.md). The model is also available on the [Model Hub](https://huggingface.co/stefan-it/byt5-small-english).
              We are now using this model as inital checkpoint to continue pretraining with the German corpus.
* 09.04.2023: Preliminary experiments on English AjMC show, that the pretrained model on English corpus is not on-par
              with current SOTA. For that reason, we are trying the Hugging Face Transformers JAX/FLAX implementation
              to pretrain models. Details can be found in [this readme](hmbyt5-flax/README.md). Pretraining has already
              started.
* 07.04.2023: Pretraining for 200k steps on the English corpus finished without crashes! TensorBoard logs can be found
              on the [Model Hub](https://huggingface.co/stefan-it/byt5-small-historic-multilingual/tensorboard). We
              also uploaded all checkpoints (we checkpoint every 25k steps)
              [here](https://huggingface.co/stefan-it/byt5-small-historic-multilingual). Fine-Tuning experiments (NER)
              on the English part of AjMC corpus from HIPE-2022 are running.
* 03.04.2023: We start ByT5 pretraining with official ByT5 implementation on a v3-32 TPU Pod - thankfully provided by
              [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC) program. Plan is to pretrain on the
              English corpus for 200k steps and use the original ByT5 Small model as init checkpoint.

# Planned/upcoming stuff

* Hyper-parameter experiments: For example pretraining English model with different learning rates for a fixed number of
  hours to compare performance (both training and eval loss).

# Acknowledgements

We thank [Luisa März](https://github.com/LuisaMaerz), [Katharina Schmid](https://github.com/schmika) and
[Erion Çano](https://github.com/erionc) for their fruitful discussions about Historic Language Models.

Research supported with Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).
Many Thanks for providing access to the TPUs ❤️
