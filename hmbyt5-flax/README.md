# hmByT5 JAX/FLAX pretraining

We use the official JAX/FLAX example in Hugging Face Transformers to pretrain a ByT5 model on a single v3-8 TPU.

The following steps are adopted from the [TPU CM Cheatsheet](https://gist.github.com/stefan-it/0a61c0625cc1f37425e9233f95332630).

# TPU VM Setup

| Library                                                     | Version |
|-------------------------------------------------------------|---------|
| [JAX](https://github.com/google/jax)                        | 0.3.25  |
| [FLAX](https://github.com/google/flax)                      | 0.6.4   | 
| [Datasets](https://github.com/huggingface/datasets)         | 2.10.1  |
| [Transformers](https://github.com/huggingface/transformers) | 4.27.1  |
| [Chex](https://github.com/deepmind/chex)                    | 0.1.6   |

Please note that it could work with later versions - but it's not guaranteed ;)

## Create disk with additional storage

```
gcloud compute disks create lms --zone us-central1-a --size 1024G
```

Make sure, that your disk is in the same `zone` as your TPU VM!

## Create v3-8 TPU VM

The following commands creates a v3-8 TPU VM and attaches the previously created disk to it:

```
gcloud alpha compute tpus tpu-vm create hmbyt5 --zone us-central1-a \
--accelerator-type v3-8 \
--version tpu-vm-base \
--data-disk source=projects/<project-name>/zones/us-central1-a/disks/lms
```

## SSH into TPU VM

Just run the following command to SSH into the TPU VM:

```
gcloud alpha compute tpus tpu-vm ssh hmbyt5 --zone us-central1-a 
```

## Installation of libraries

After ssh'ing into TPU VM, run the following commands in e.g. `tmux`.

```
sudo apt update -y && sudo apt install -y python3-venv
python3 -m venv $HOME/dev
source $HOME/dev/bin/activate
pip install "jax[tpu]==0.3.25" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install ipython requests
git clone https://github.com/huggingface/transformers.git
git clone https://github.com/huggingface/datasets.git
git clone https://github.com/google/flax.git
cd transformers && git checkout v4.27.1 && pip3 install -e . && cd ..
cd datasets && git checkout 2.10.1 && pip3 install -e . && cd ..
cd flax && git checkout v0.6.4 && pip3 install -e . && cd ..
pip install chex==0.1.6

# Useful symlinks
ln -s $HOME/transformers/examples/flax/language-modeling/run_t5_mlm_flax.py run_t5_mlm_flax.py
```

## Format and mount disk

The attached disk needs to formatted first using:

```
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
```

After that it can be mounted via:

```
sudo mkdir -p /mnt/datasets
sudo mount -o discard,defaults /dev/sdb /mnt/datasets/
sudo chmod a+w /mnt/datasets
```

## HF Datasets Cache

The HF dataset cache variable should now point to the mounted disk:

```
export HF_DATASETS_CACHE=/mnt/datasets/huggingface
```

## Create swapfile

The following commands create and activate a swapfile:

```
cd /mnt/datasets
sudo fallocate -l 10GB ./swapfile
sudo chmod 600 ./swapfile
sudo mkswap ./swapfile
sudo swapon ./swapfile
```

## TensorBoard

Install TensorBoard to get better training metric visualizations:

```
pip install tensorboard==2.12.1 tensorflow==2.12
```

## Hugging Face Model Hub Login

In order to push all checkpoints directly to the Model Hub, we need to setup Git-LFS first:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install -y git-lfs
git lfs install
git config --global credential.helper store
```

After that, Model Hub credentials need to be stored:

```bash
huggingface-cli login
```

# Validation Data

We use available training splits from NER corpora to construct a validation dataset.
The [`create_validation_data.py`](create_validation_data.py) can be used to create validation splits for all languages.

# Pretraining

## English

In our first experiment, we train for one epoch over the English (blbooks) corpus, using the following command:

```bash
python run_t5_mlm_flax.py \
--model_name_or_path="google/byt5-small" \
--output_dir="/mnt/datasets/byt5-small-english" \
--max_seq_length="1024" \
--per_device_train_batch_size="16" \
--per_device_eval_batch_size="16" \
--learning_rate="0.0003" \
--warmup_steps="10000" \
--logging_steps="500" \
--save_steps="10000" \
--eval_steps="2500" \
--train_file="/mnt/datasets/corpora/english.txt" \
--validation_file="/mnt/datasets/validation/en_validation.txt" \
--hub_model_id="stefan-it/byt5-small-english" \
--num_train_epochs="1" \
--preprocessing_num_workers="16" \
--push_to_hub
```

Checkpoints and the TensorBoard logs are automatically uploaded to the Model Hub, and can be found
[here](https://huggingface.co/stefan-it/byt5-small-english).
