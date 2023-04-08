from flair.data import MultiCorpus
from flair.datasets import NER_HIPE_2022, NER_ICDAR_EUROPEANA

validation_datasets = {
    "en": ["ajmc", "topres19th"],
    "de": ["ajmc", "hipe2020", "newseye"],
    "fr": ["ajmc", "hipe2020", "letemps"],
    "fi": ["newseye"],
    "sv": ["newseye"],
    "nl": ["icdar-europeana"],
}

limit_tokens = 176_183

for language, datasets in validation_datasets.items():
    corpora = []
    for dataset_name in datasets:
        if dataset_name == "icdar-europeana":
            corpus = NER_ICDAR_EUROPEANA(language="nl")
        else:
            corpus = NER_HIPE_2022(dataset_name=dataset_name, language=language, add_document_separator=False)

        corpora.append(corpus)

    corpora = MultiCorpus(corpora)

    token_counter = 0

    with open(f"{language}_validation.txt", "wt") as f_p:
        for sentence in corpora.train:
            tokens = sentence.tokens
            token_counter += len(tokens)

            if token_counter > limit_tokens:
                continue
            f_p.write(" ".join(token.text for token in tokens) + "\n")

    print(f"Number of tokens for {language}: {token_counter}")
