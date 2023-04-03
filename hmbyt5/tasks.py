import functools

import seqio
import t5.data
import t5.data.tasks
from t5.evaluation import metrics

MEAN_NOISE_SPAN_LENGTH = 20

DEFAULT_PREPROCESSORS = [
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]

DEFAULT_BYTE_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(vocabulary=t5.data.ByteVocabulary()),
    "targets": t5.data.Feature(vocabulary=t5.data.ByteVocabulary())
}

FEATURE_MAP = {
    "byt5": DEFAULT_BYTE_OUTPUT_FEATURES,
}

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

seqio.TaskRegistry.add(
    "de_corpus",
    source=seqio.TfdsDataSource(tfds_name="de_dataset:1.0.0"),
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

seqio.TaskRegistry.add(
    "fr_corpus",
    source=seqio.TfdsDataSource(tfds_name="fr_dataset:1.0.0"),
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

seqio.TaskRegistry.add(
    "fi_corpus",
    source=seqio.TfdsDataSource(tfds_name="fi_dataset:1.0.0"),
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

seqio.TaskRegistry.add(
    "sv_corpus",
    source=seqio.TfdsDataSource(tfds_name="sv_dataset:1.0.0"),
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

seqio.TaskRegistry.add(
    "nl_corpus",
    source=seqio.TfdsDataSource(tfds_name="nl_dataset:1.0.0"),
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

# We upsampled all language with our own upsampling logic.
# So we can use a rate of 1.0 for each language here!
seqio.MixtureRegistry.add(
  "hm_corpus",
  ["en_corpus", "de_corpus", "fr_corpus", "fi_corpus", "sv_corpus", "nl_corpus"],
  default_rate=1
)
