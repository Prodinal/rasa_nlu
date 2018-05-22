import spacy
from spacy.tokens import Doc
import urllib
import xml.etree.ElementTree as ET
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.components import Component


class MyTokenizer():
    def __init__(self, vocab):
        self.vocab = vocab
        self.url = 'http://localhost:8000/process?run=QT&text='

    def __call__(self, input):
        text = urllib.parse.quote_plus(input)

        result = urllib.request.urlopen(self.url + text).read()
        annotations = ET.fromstring(result).find('AnnotationSet')

        words = [element.getchildren()[1].find('Value').text
                 for element in annotations.getchildren()
                 if element.get('Type') == 'Token']
        return Doc(self.vocab, words=words)


class HuTokenizer(Tokenizer, Component):
    name = "tokenizer_hu"
    provides = ["tokens", "spacy_doc"]

    def __init__(self):
        self.vocab = spacy.blank('en').vocab
        self.url = 'http://localhost:8000/process?run=QT&text='

    def process(self, message, **kwargs):
        print('Processing text: ' + message.text)
        # type: (Message, **Any) -> None
        doc = self.tokenize(message.text)
        message.set("spacy_doc", doc)
        message.set("tokens", [Token(t.text, t.idx) for t in doc])

    def tokenize(self, text):
        text = urllib.parse.quote_plus(text)

        result = urllib.request.urlopen(self.url + text).read()
        annotations = ET.fromstring(result).find('AnnotationSet')

        words = [element.getchildren()[1].find('Value').text
                 for element in annotations.getchildren()
                 if element.get('Type') == 'Token']
        return Doc(self.vocab, words=words)


if __name__ == "__main__":
    nlp = spacy.load('en')
    nlp.tokenizer = MyTokenizer(nlp.vocab)
    doc = nlp('Jó, hogy ez az alma piros, mert az olyan almákat szeretem.')
    for token in doc:
        print('Token is: ' + str(token))
