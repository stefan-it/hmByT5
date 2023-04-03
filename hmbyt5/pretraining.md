# hmByT5 Pretraining

This section documents pretraining of a ByT5 model using the original T5 and ByT5 code base.

# Preparing VM

In the next step, a VM needs to be created to coordinate the pre-training process. We are using a `n1-standard-2` instance and a customized boot disk size of 50GB. Please notice that the default boot disk size is 10GB, which is not enough for all Python dependencies. We are using TensorFlow 2.8 in our experiments:

```bash
$ gcloud compute instances create byt5 --zone=europe-west4-a \
  --machine-type=n1-standard-2 --image-project=ml-images \
  --image-family=tf-2-8-0 --scopes=cloud-platform \
  --boot-disk-size 50GB
```

The VM should be in the same zone as GCP bucket and TPU.

After VM creation, we can SSH into it:

```bash
$ gcloud compute ssh byt5 --zone europe-west4-a
```

Now we immediately start a `tmux` session and run all commands in this session. If the connection to the VM got lost, you can re-sume the session with `tmux attach` after next login.

## Installing Dependencies

We just need to clone the T5 repository:

```bash
$ git clone https://github.com/google-research/text-to-text-transfer-transformer.git
$ cd text-to-text-transfer-transformer
$ git checkout c3be7cf
$ pip3 install -e .
$ export PATH=$PATH:$HOME/.local/bin
```

Note: we need to use a special commit, because of recent code changes that are not compatible with Python 3.7. The following dependencies needs a downgrade:

```bash
$ pip3 install --upgrade pyglove==0.2.1
$ pip3 install seqio==0.0.13
```

This will install all necessary dependencies. To make sure that everything is working, just run:

```bash
$ t5_mesh_transformer --helpfull
```

In the next step, the ByT5 repo is cloned:

```bash
$ git clone https://github.com/google-research/byt5.git
$ cd byt5
```

Clone this repository into the ByT5 repo, so that our custom tasks can be used:

```bash
$ git clone https://github.com/stefan-it/hmByT5.git
```

## Custom SeqIO Tasks

To pretrain a ByT5 model on our own corpus, we need to slightly extend the ByT5 library. To do so, we add our datasets to the internal task registry.
This is done in the `hmbyt5/tasks.py` file. Here's an example of the English dataset:

```python
seqio.TaskRegistry.add(
    "en_corpus",
    source=seqio.TfdsDataSource(tfds_name="en_dataset:1.0.0"),
    preprocessors=[
          functools.partial(
              t5.data.preprocessors.rekey,
              key_map={
                  "inputs": None,
                  "targets": "text"
              }),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          functools.partial(t5.data.preprocessors.span_corruption,
                            mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH),
          seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_BYTE_OUTPUT_FEATURES,
    metric_fns=[]
)
```

## Model Configurations

Our current strategy is using the original ByT5 (Small) Model as init checkpoint. Then we continue pretraining language-after-language: 100k steps for each language.
Thus, 6 different GIN configuration files are located in the `configs` folder:

* `./configs/0_english_operative_config.gin`
