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