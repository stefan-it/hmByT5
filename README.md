# hmByT5

Upcoming Historic Multilingual ByT5 Model. It covers the following languages:

* English (British Library Corpus - Books)
* German (Europeana Newspaper)
* French (Europeana Newspaper)
* Finnish (Europeana Newspaper)
* Swedish (Europeana Newspaper)
* Dutch (Delpher Corpus)

# Pretraining

We pretrain hmByT5 on a v3-32 TPU Pod. Details about the training can be found [here](hmbyt5/README.md).

# Evaluation on Downstream Tasks (NER)

We use Flair to fine-tune hmByT5 on HIPE-2022 data. Details about the fine-tuning can be found [here](bench/README.md).

# Acknowledgements

Research supported with Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).
Many Thanks for providing access to the TPUs ❤️
