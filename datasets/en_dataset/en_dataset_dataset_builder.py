"""en_dataset dataset."""

import tensorflow_datasets as tfds

from pathlib import Path

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for de_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
      features=tfds.features.FeaturesDict({
        'text': tfds.features.Text(),
      }),
      supervised_keys=None,  # Set to `None` to disable
      homepage='https://github.com/stefan-it/hmByT5',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {
      'train': self._generate_examples("/home/stefan/hmbyt5/upsampled_data"),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    cnt = 0

    for file in Path(path).iterdir():
        if not file.name.endswith("english.txt"):
          continue

        print(f"Reading file: {file}...")

        with open(file, "rt") as f_p:
          for line in f_p:
            line = line.rstrip()

            if not line:
              continue
            yield cnt, {
              'text': line,
            }
            cnt += 1
