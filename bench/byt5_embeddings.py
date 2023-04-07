import os
import flair
import torch

from flair.data import Sentence
from flair.embeddings import TokenEmbeddings
from flair.embeddings.base import register_embeddings
from flair.embeddings.transformer import TransformerBaseEmbeddings
from transformers import T5EncoderModel, ByT5Tokenizer, AutoConfig, PretrainedConfig

from io import BytesIO

from typing import Dict, List, Union, Optional

@register_embeddings
class ByT5Embeddings(TransformerBaseEmbeddings):
    def __init__(
        self,
        model: str = "google/byt5-base",
        fine_tune: bool = True,
        layers: str = "-1",
        layer_mean: bool = True,
        subtoken_pooling: str = "first",
        cls_pooling: str = "cls",
        is_token_embedding: bool = True,
        is_document_embedding: bool = False,
        allow_long_sentences: bool = False,
        use_context: Union[bool, int] = False,
        respect_document_boundaries: bool = False,
        context_dropout: float = 0.0,
        saved_config: Optional[PretrainedConfig] = None,
        tokenizer_data: Optional[BytesIO] = None,
        feature_extractor_data: Optional[BytesIO] = None,
        name: Optional[str] = None,
        force_max_length: bool = False,
        needs_manual_ocr: Optional[bool] = None,
        use_context_separator: bool = True,
        **kwargs,
    ):
        self.instance_parameters = self.get_instance_parameters(locals=locals())
        del self.instance_parameters["saved_config"]
        del self.instance_parameters["tokenizer_data"]
        # temporary fix to disable tokenizer parallelism warning
        # (see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # do not print transformer warnings as these are confusing in this case
        from transformers import logging

        logging.set_verbosity_error()

        self.tokenizer: PreTrainedTokenizer
        self.feature_extractor: Optional[FeatureExtractionMixin]

        if tokenizer_data is None:
            # load tokenizer and transformer model
            self.tokenizer = ByT5Tokenizer(model)
            self.feature_extractor = None
        else:
            # load tokenizer from inmemory zip-file
            self.tokenizer = self._tokenizer_from_bytes(tokenizer_data)
            self.feature_extractor = None

        if saved_config is None:
            config = AutoConfig.from_pretrained(model, output_hidden_states=True, **kwargs)
            from transformers import T5EncoderModel
            transformer_model = T5EncoderModel.from_pretrained(model, config=config)
        else:
            from transformers import T5EncoderModel
            transformer_model = T5EncoderModel(saved_config, **kwargs)

        transformer_model = transformer_model.to(flair.device)

        self.truncate = True
        self.force_max_length = force_max_length

        allow_long_sentences = False
        self.truncate = False

        self.stride = self.tokenizer.model_max_length // 2 if allow_long_sentences else 0
        self.allow_long_sentences = allow_long_sentences
        self.use_lang_emb = hasattr(transformer_model, "use_lang_emb") and transformer_model.use_lang_emb

        # model name
        if name is None:
            self.name = "transformer-" + transformer_model.name_or_path
        else:
            self.name = name
        self.base_model_name = transformer_model.name_or_path

        self.token_embedding = is_token_embedding
        self.document_embedding = is_document_embedding

        if self.token_embedding and subtoken_pooling not in ["first", "last", "first_last", "mean"]:
            raise ValueError(f"Subtoken Pooling operation `{subtoken_pooling}` is not defined for TransformerEmbedding")

        self.context_length = 0

        self.context_dropout = context_dropout
        self.respect_document_boundaries = respect_document_boundaries

        # embedding parameters
        if layers == "all":
            # send mini-token through to check how many layers the model has
            input_ids = torch.tensor([1], device=flair.device).unsqueeze(0)
            hidden_states = self.model(input_ids).hidden_states
            self.layer_indexes = [int(x) for x in range(len(hidden_states))]
        else:
            self.layer_indexes = [int(x) for x in layers.split(",")]

        self.cls_pooling = cls_pooling
        self.subtoken_pooling = subtoken_pooling
        self.layer_mean = layer_mean
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune

        # return length
        self.embedding_length_internal = self._calculate_embedding_length(transformer_model)
        self.needs_manual_ocr = False

        # If we use a context separator, add a new special token
        self.use_context_separator = False

        super().__init__(**self.to_args())

        # most models have an initial BOS token, except for XLNet, T5 and GPT2
        self.initial_cls_token: bool = self._has_initial_cls_token()

        self.model = transformer_model

        self.to(flair.device)
        # when initializing, embeddings are in eval mode by default
        self.eval()

    @property
    def embedding_length(self) -> int:
        if not hasattr(self, "embedding_length_internal"):
            self.embedding_length_internal = self._calculate_embedding_length(self.model)

        return self.embedding_length_internal

    def _has_initial_cls_token(self) -> bool:
        # most models have CLS token as last token (GPT-1, GPT-2, TransfoXL, XLNet, XLM), but BERT is initial
        tokens = self.tokenizer.encode("a")
        return tokens[0] == self.tokenizer.cls_token_id

    def _calculate_embedding_length(self, model) -> int:
        if not self.layer_mean:
            length = len(self.layer_indexes) * model.config.hidden_size
        else:
            length = model.config.hidden_size

        # in case of doubt: token embedding has higher priority than document embedding
        if self.token_embedding and self.subtoken_pooling == "first_last":
            length *= 2
        return length

    @property
    def embedding_type(self) -> str:
        # in case of doubt: token embedding has higher priority than document embedding
        return "word-level" if self.token_embedding else "sentence-level"

    def __setstate__(self, state):
        config_state_dict = state.pop("config_state_dict", None)
        model_state_dict = state.pop("model_state_dict", None)

        # legacy TransformerDocumentEmbedding
        state.pop("batch_size", None)
        state.pop("embedding_length_internal", None)
        # legacy TransformerTokenEmbedding
        state.pop("memory_effective_training", None)

        if "base_model_name" in state:
            state["model"] = state.pop("base_model_name")

        state["use_context"] = state.pop("context_length", False)

        if "layer_indexes" in state:
            layer_indexes = state.pop("layer_indexes")
            state["layers"] = ",".join(map(str, layer_indexes))

        if "use_context_separator" not in state:
            # legacy Flair <= 0.12
            state["use_context_separator"] = False

        if "use_scalar_mix" in state:
            # legacy Flair <= 0.7
            state["layer_mean"] = state.pop("use_scalar_mix")

        if "is_token_embedding" not in state:
            # legacy TransformerTokenEmbedding
            state["is_token_embedding"] = "pooling_operation" in state

        if "is_document_embedding" not in state:
            # Legacy TransformerDocumentEmbedding
            state["is_document_embedding"] = "pooling" in state

        if "pooling_operation" in state:
            # legacy TransformerTokenEmbedding
            state["subtoken_pooling"] = state.pop("pooling_operation")

        if "pooling" in state:
            # legacy TransformerDocumentEmbedding
            state["cls_pooling"] = state.pop("pooling")

        config = None

        if config_state_dict:
            model_type = config_state_dict.get("model_type", "bert")
            config_class = CONFIG_MAPPING[model_type]
            config = config_class.from_dict(config_state_dict)

        embedding = self.create_from_state(saved_config=config, **state)

        # copy values from new embedding
        for key in embedding.__dict__.keys():
            self.__dict__[key] = embedding.__dict__[key]

        if model_state_dict:
            self.model.load_state_dict(model_state_dict)

    @classmethod
    def from_params(cls, params):
        params["use_context"] = params.pop("context_length", 0)
        return cls.create_from_state(**params)

    def to_params(self):
        config_dict = self.model.config.to_dict()
        super_params = super().to_params()

        model_state = {
            **super_params,
            "model": self.base_model_name,
            "fine_tune": self.fine_tune,
            "layers": ",".join(map(str, self.layer_indexes)),
            "layer_mean": self.layer_mean,
            "subtoken_pooling": self.subtoken_pooling,
            "cls_pooling": self.cls_pooling,
            "config_state_dict": config_dict,
        }

        return model_state

    def _can_document_embedding_shortcut(self):
        # cls first pooling can be done without recreating sentence hidden states
        return (
            self.document_embedding
            and not self.token_embedding
            and self.cls_pooling == "cls"
            and self.initial_cls_token
        )

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for sentence in sentences:
            encoding = self.tokenizer([sentence.to_tokenized_string()], padding="longest", return_tensors="pt")

            # Pass to model
            encoding["input_ids"] = encoding["input_ids"].to(flair.device)
            encoding["attention_mask"] = encoding["attention_mask"].to(flair.device)
            
            if self.fine_tune:
                hidden_states = self.model(**encoding).hidden_states
            else:
                with torch.no_grad():
                    hidden_states = self.model(**encoding).hidden_states

            offset = 0

            for token in sentence.tokens:
                token_length = len(self.tokenizer([token.text], padding="longest", return_tensors="pt")["input_ids"][0]) - 1

                token_embeddings: List[torch.FloatTensor] = []

                for layer_index in self.layer_indexes:
                    current_character_embeddings = hidden_states[layer_index][0][offset: offset + token_length]

                    if self.subtoken_pooling == "first":
                        final_character_embedding = current_character_embeddings[0]

                    if self.subtoken_pooling == "last":
                        final_character_embedding = current_character_embeddings[-1]

                    if self.subtoken_pooling == "first_last":
                        final_character_embedding = torch.cat([current_character_embeddings[0],
                                                               current_character_embeddings[-1]])

                    if self.subtoken_pooling == "mean":
                        all_embeddings: List[torch.FloatTensor] = [
                            embedding.unsqueeze(0) for embedding in current_character_embeddings
                        ]
                        final_character_embedding = torch.mean(torch.cat(all_embeddings, dim=0), dim=0)

                    token_embeddings.append(final_character_embedding)

                if self.layer_mean:
                    mean_token_embeddings = torch.mean(torch.stack(token_embeddings, dim=1), dim=1)
                    token_embeddings = [mean_token_embeddings]

                token.set_embedding(self.name, torch.cat(token_embeddings))

                offset += token_length + 1
                
    def _forward_tensors(self, tensors) -> Dict[str, torch.Tensor]:
        return self.forward(**tensors)

