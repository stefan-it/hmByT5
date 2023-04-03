# TensorFlow Dataset Creation

TensorFlow 2.12.0 and TensorFlow datasets in version 4.8.3 are used in our experiments.

For each language, we create TensorFlow Datasets via:

```bash
$ tfds new <language>_dataset
```

More precisely, six datasets will be created:

```bash
$ tfds new en_dataset
$ tfds new de_dataset
$ tfds new fr_dataset
$ tfds new fi_dataset
$ tfds new sv_dataset
$ tfds new nl_dataset
```

Then each dataset will be configured to read-in the original corpus for the corresponding language.

## TensorFlow Dataset Stats

After that, the TensorFlow Dataset can be built for each language:

```bash
$ tfds build
```

After dataset creation has completed, datasets stats are returned.

### English

```python
tfds.core.DatasetInfo(
    name='en_dataset',
    full_name='en_dataset/1.0.0',
    description="""
    TODO(en_dataset): Markdown description of that will appear on the catalog page.
    Description is **formatted** as markdown.

    It should also contain any processing which has been applied (if any),
    (e.g. corrupted example skipped, images cropped,...):
    """,
    homepage='https://github.com/stefan-it/hmByT5',
    data_path='/home/stefan/tensorflow_datasets/en_dataset/1.0.0',
    file_format=tfrecord,
    download_size=Unknown size,
    dataset_size=27.60 GiB,
    features=FeaturesDict({
        'text': Text(shape=(), dtype=string),
    }),
    supervised_keys=None,
    disable_shuffling=True,
    splits={
        'train': <SplitInfo num_examples=16501636, num_shards=256>,
    },
    citation="""// TODO(en_dataset): BibTeX citation""",
)

```

### German

```python
tfds.core.DatasetInfo(
    name='de_dataset',
    full_name='de_dataset/1.0.0',
    description="""
    TODO(de_dataset): Markdown description of that will appear on the catalog page.
    Description is **formatted** as markdown.

    It should also contain any processing which has been applied (if any),
    (e.g. corrupted example skipped, images cropped,...):
    """,
    homepage='https://github.com/stefan-it/hmByT5',
    data_path='/home/stefan/tensorflow_datasets/de_dataset/1.0.0',
    file_format=tfrecord,
    download_size=Unknown size,
    dataset_size=28.23 GiB,
    features=FeaturesDict({
        'text': Text(shape=(), dtype=string),
    }),
    supervised_keys=None,
    disable_shuffling=True,
    splits={
        'train': <SplitInfo num_examples=60142681, num_shards=256>,
    },
    citation="""// TODO(de_dataset): BibTeX citation""",
)
```

### French

```python
tfds.core.DatasetInfo(
    name='fr_dataset',
    full_name='fr_dataset/1.0.0',
    description="""
    TODO(fr_dataset): Markdown description of that will appear on the catalog page.
    Description is **formatted** as markdown.

    It should also contain any processing which has been applied (if any),
    (e.g. corrupted example skipped, images cropped,...):
    """,
    homepage='https://github.com/stefan-it/hmByT5',
    data_path='/home/stefan/tensorflow_datasets/fr_dataset/1.0.0',
    file_format=tfrecord,
    download_size=Unknown size,
    dataset_size=28.08 GiB,
    features=FeaturesDict({
        'text': Text(shape=(), dtype=string),
    }),
    supervised_keys=None,
    disable_shuffling=True,
    splits={
        'train': <SplitInfo num_examples=46825943, num_shards=256>,
    },
    citation="""// TODO(fr_dataset): BibTeX citation""",
)
```

### Finnish

```python
tfds.core.DatasetInfo(
    name='fi_dataset',
    full_name='fi_dataset/1.0.0',
    description="""
    TODO(fi_dataset): Markdown description of that will appear on the catalog page.
    Description is **formatted** as markdown.

    It should also contain any processing which has been applied (if any),
    (e.g. corrupted example skipped, images cropped,...):
    """,
    homepage='https://github.com/stefan-it/hmByT5',
    data_path='/home/stefan/tensorflow_datasets/fi_dataset/1.0.0',
    file_format=tfrecord,
    download_size=Unknown size,
    dataset_size=28.53 GiB,
    features=FeaturesDict({
        'text': Text(shape=(), dtype=string),
    }),
    supervised_keys=None,
    disable_shuffling=True,
    splits={
        'train': <SplitInfo num_examples=77362399, num_shards=256>,
    },
    citation="""// TODO(fi_dataset): BibTeX citation""",
)
```

### Swedish

```python
tfds.core.DatasetInfo(
    name='sv_dataset',
    full_name='sv_dataset/1.0.0',
    description="""
    TODO(sv_dataset): Markdown description of that will appear on the catalog page.
    Description is **formatted** as markdown.

    It should also contain any processing which has been applied (if any),
    (e.g. corrupted example skipped, images cropped,...):
    """,
    homepage='https://github.com/stefan-it/hmByT5',
    data_path='/home/stefan/tensorflow_datasets/sv_dataset/1.0.0',
    file_format=tfrecord,
    download_size=Unknown size,
    dataset_size=28.19 GiB,
    features=FeaturesDict({
        'text': Text(shape=(), dtype=string),
    }),
    supervised_keys=None,
    disable_shuffling=True,
    splits={
        'train': <SplitInfo num_examples=54114695, num_shards=256>,
    },
    citation="""// TODO(sv_dataset): BibTeX citation""",
)
```

### Dutch

```python
tfds.core.DatasetInfo(
    name='nl_dataset',
    full_name='nl_dataset/1.0.0',
    description="""
    TODO(nl_dataset): Markdown description of that will appear on the catalog page.
    Description is **formatted** as markdown.

    It should also contain any processing which has been applied (if any),
    (e.g. corrupted example skipped, images cropped,...):
    """,
    homepage='https://github.com/stefan-it/hmByT5',
    data_path='/home/stefan/tensorflow_datasets/nl_dataset/1.0.0',
    file_format=tfrecord,
    download_size=Unknown size,
    dataset_size=34.53 GiB,
    features=FeaturesDict({
        'text': Text(shape=(), dtype=string),
    }),
    supervised_keys=None,
    disable_shuffling=True,
    splits={
        'train': <SplitInfo num_examples=551904949, num_shards=512>,
    },
    citation="""// TODO(nl_dataset): BibTeX citation""",
)
```