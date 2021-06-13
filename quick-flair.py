from flair.models import SequenceTagger, TextClassifier
from flair.data import Sentence

tagger = SequenceTagger.load('ner-ontonotes-fast')
classifier = TextClassifier.load('sentiment-fast')

texts = [
    "Today is quite an exciting day",
    "George Washington went to Washington.",
    "enormously entertaining for moviegoers of any age.",
    "Terrible!!!",
    "Nothing to say",
    "I'm bored",
    "I rather do not say my opinion"
    "I love my iPhone"
    ]

sentences = [Sentence(text) for text in texts]

tagger.predict(sentences)
classifier.predict(sentences)

for sentence in sentences:
    print(sentence.annotation_layers)
    print(sentence.get_spans())
    print(sentence.get_labels())

