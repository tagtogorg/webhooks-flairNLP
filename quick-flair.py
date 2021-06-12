from flair.models import SequenceTagger
from flair.data import Sentence

tagger = SequenceTagger.load('ner-ontonotes-fast')

sentence = Sentence('George Washington went to Washington.')

# predict NER tags
tagger.predict(sentence)

# print sentence with predicted tags
print(sentence.to_tagged_string())

for entity in sentence.get_spans('ner'):
    print(entity)

print(sentence.to_dict(tag_type='ner'))
