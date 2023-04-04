import click
import json
import logging
import sys

import flair
import torch

from typing import List

from flair.data import MultiCorpus
from flair.datasets import ColumnCorpus, NER_HIPE_2022
from flair.embeddings import (
    TokenEmbeddings,
    StackedEmbeddings,
    TransformerWordEmbeddings
)
from flair import set_seed
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from utils import prepare_ajmc_corpus

logger = logging.getLogger("flair")
logger.setLevel(level="INFO")


def run_experiment(seed: int, batch_size: int, epoch: int, learning_rate: float, subword_pooling: str, hipe_datasets: List[str], json_config: dict):
    hf_model = json_config["hf_model"]
    context_size = json_config["context_size"]
    layers = json_config["layers"] if "layers" in json_config else "-1"
    use_crf = json_config["use_crf"] if "use_crf" in json_config else False

    # Set seed for reproducibility
    set_seed(seed)

    corpus_list = [] 

    # Dataset-related
    for dataset in hipe_datasets:
        dataset_name, language = dataset.split("/")

        preproc_fn = None

        if dataset_name == "ajmc":
            preproc_fn = prepare_ajmc_corpus

        corpus_list.append(NER_HIPE_2022(dataset_name=dataset_name, language=language, preproc_fn=preproc_fn, add_document_separator=True))

    if context_size == 0:
        context_size = False

    logger.info("FLERT Context:", context_size)
    logger.info("Layers:", layers)
    logger.info("Use CRF:", use_crf)

    corpora: MultiCorpus = MultiCorpus(corpora=corpus_list, sample_missing_splits=False)
    label_dictionary = corpora.make_label_dictionary(label_type="ner")
    logger.info("Label Dictionary:", label_dictionary.get_items())

    # Embeddings
    embeddings = TransformerWordEmbeddings(
        model=hf_model,
        layers=layers,
        subtoken_pooling="first",
        fine_tune=True,
        subword_pooling=subword_pooling,
        use_context=context_size,
    )

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dictionary,
        tag_type="ner",
        use_crf=use_crf,
        use_rnn=False,
        reproject_embeddings=False,
    )

    # Trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpora)

    datasets = "-".join([dataset for dataset in hipe_datasets])

    trainer.fine_tune(
        f"hipe2022-{datasets}-{hf_model}-bs{batch_size}-ws{context_size}-e{epoch}-lr{learning_rate}-pooling{subword_pooling}-layers{layers}-crf{use_crf}-{seed}",
        learning_rate=learning_rate,
        mini_batch_size=batch_size,
        max_epochs=epoch,
        shuffle=True,
        embeddings_storage_mode='none',
        weight_decay=0.,
        use_final_model_for_eval=False,
    )
    
    # Finally, print model card for information
    tagger.print_model_card()


if __name__ == "__main__":
    filename = sys.argv[1]
    with open(filename, "rt") as f_p:
        json_config = json.load(f_p)

    seeds = json_config["seeds"]
    batch_sizes = json_config["batch_sizes"]
    epochs = json_config["epochs"]
    learning_rates = json_config["learning_rates"]
    subword_poolings = json_config["subword_poolings"]

    hipe_datasets = json_config["hipe_datasets"] # Do not iterate over them

    cuda = json_config["cuda"]
    flair.device = f'cuda:{cuda}'

    for seed in seeds:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for learning_rate in learning_rates:
                    for subword_pooling in subword_poolings:
                        run_experiment(seed, batch_size, epoch, learning_rate, subword_pooling, hipe_datasets, json_config)  # pylint: disable=no-value-for-parameter
