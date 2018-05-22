from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

# Spacy Hu components
from SpacyHu import (
    Tokenizer,
    ConstitutencyParser,
    DependencyParser,
    LemmatizerMorphAnalyzer,
    POSTagger,
    NPChunker,
    PreverbIdentifier,
    HuWordToVec
)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from rasa_nlu.model import Metadata


class SpacyHuNLP(Component):
    name = "nlp_spacy_hu"

    provides = ["spacy_doc", "spacy_nlp"]

    def __init__(self, nlp, gate_server_url):
        # type: (Language, Text, Text) -> None

        self.nlp = nlp
        self.gate_server_url = gate_server_url

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["spacy"]

    @classmethod
    def create(cls, config):
        # type: (RasaNLUConfig) -> SpacyNLP

        return cls(cls.create_nlp(config["gate_server_url"]), config["gate_server_url"])

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "gate_server_url": self.gate_server_url,
        }

    @classmethod
    def load(cls,
             model_dir=None,
             model_metadata=None,
             cached_component=None,
             **kwargs):
        # type: (Text, Metadata, Optional[SpacyNLP], **Any) -> SpacyNLP
        import spacy

        if cached_component:
            return cached_component

        return cls(cls.create_nlp(model_metadata.get("gate_server_url")), model_metadata.get("gate_server_url"))


    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Text

        spacy_model_name = model_metadata.metadata.get("spacy_model_name")
        if spacy_model_name is None:
            # Fallback, use the language name, e.g. "en",
            # as the model name if no explicit name is defined
            spacy_model_name = model_metadata.language
        return cls.name + "-" + spacy_model_name

    def provide_context(self):
        # type: () -> Dict[Text, Any]

        return {"spacy_nlp": self.nlp}

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("spacy_doc", self.nlp(example.text.lower()))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("spacy_doc", self.nlp(message.text.lower()))

    @staticmethod
    def create_nlp(gate_server_url):
        import spacy
        print(gate_server_url)
        nlp = spacy.blank('hu')
        nlp.tokenizer = Tokenizer.HuTokenizer(nlp.vocab, url=gate_server_url)

        morph_analyzer = LemmatizerMorphAnalyzer.HuLemmaMorph(nlp, url=gate_server_url)
        nlp.add_pipe(morph_analyzer)

        constitutency_parser = ConstitutencyParser.ConstitutencyParser(nlp, url=gate_server_url)
        nlp.add_pipe(constitutency_parser)

        dependency_parser = DependencyParser.DependencyParser(nlp, url=gate_server_url)
        nlp.add_pipe(dependency_parser)

        np_chunker = NPChunker.NPChunker(nlp, url=gate_server_url)
        nlp.add_pipe(np_chunker)

        POS_analyzer = POSTagger.HuPOSTagger(nlp, url=gate_server_url)
        nlp.add_pipe(POS_analyzer)

        preverb_identifier = PreverbIdentifier.PreverbIdentifier(nlp, url=gate_server_url)
        nlp.add_pipe(preverb_identifier)

        hu_word_to_vec = HuWordToVec.HUWordToVec()
        nlp.add_pipe(hu_word_to_vec)

        return nlp
