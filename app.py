from flair.data import Label
from flask import Flask, request
import requests
import os
import json
from bs4 import BeautifulSoup
from flair.models import SequenceTagger, TextClassifier
# from flair.data import Sentence
from flair.tokenization import SegtokSentenceSplitter
from typing import Optional, Dict, Any, List, Tuple, Union

# -----------------------------------------------------------------------------

# Set your tagtog credentials & project info
MY_USERNAME = os.environ['MY_TAGTOG_USERNAME']
MY_PASSWORD = os.environ['MY_TAGTOG_PASSWORD']
MY_PROJECT = os.environ['MY_TAGTOG_PROJECT']
# the project owner could be a different user, but for simplicity we assume it's the same as your username
MY_PROJECT_OWNER = os.environ.get('MY_PROJECT_OWNER', MY_USERNAME)

TAGTOG_DOMAIN_CLOUD = "https://tagtog.net"
TAGTOG_DOMAIN = os.environ.get('TAGTOG_DOMAIN', TAGTOG_DOMAIN_CLOUD)
# When this is false, the SSL certification will not be verified (this is useful, for instance, for self-signed localhost tagtog instances)
VERIFY_SSL_CERT = (TAGTOG_DOMAIN == TAGTOG_DOMAIN_CLOUD)
MY_PROJECT_URL = f"{TAGTOG_DOMAIN}/{MY_PROJECT_OWNER}/{MY_PROJECT}"

# -----------------------------------------------------------------------------

# API authentication
auth = requests.auth.HTTPBasicAuth(username=MY_USERNAME, password=MY_PASSWORD)

tagtog_docs_API_endpoint = f"{TAGTOG_DOMAIN}/-api/documents/v1"
tagtog_sets_API_endpoint = f"{TAGTOG_DOMAIN}/-api/settings/v1"

default_API_params = {'owner': MY_PROJECT_OWNER, 'project': MY_PROJECT}

# Parameters for the POST API call to import a pre-annotated document
# (see https://docs.tagtog.net/API_documents_v1.html#import-annotated-documents-post)
post_params_doc = {**default_API_params, **{'output': 'null', 'format': 'anndoc'}}

# -----------------------------------------------------------------------------

# https://huggingface.co/flair/ner-english-ontonotes-fast
tagger_name = 'ner-ontonotes-fast'
TAGTOG_TAGGER_WHO = f'ml:flair-{tagger_name}'

classifier_name = 'sentiment-fast'
TAGTOG_CLASSIFIER_WHO = f'ml:flair-{classifier_name}'

DOC_LABEL_SENTIMENT_NAME = 'SENTIMENT'

# -----------------------------------------------------------------------------

# See: https://docs.tagtog.net/API_settings_v1.html#annotations-legend
def get_tagtog_anntasks_json_map():
  res = requests.get(f"{tagtog_sets_API_endpoint}/annotationsLegend", params=default_API_params, auth=auth, verify=VERIFY_SSL_CERT)
  assert res.status_code == 200, f"Couldn't connect to the given tagtog project with the given credentials (http status code {res.status_code}; body: {res.text})"
  return res.json()

# In the example of https://github.com/tagtog/demo-webhooks, we could hardcode this like:
# map_ids_to_names = {'e_1': 'PERSON', 'e_2': 'ORG', 'e_3': 'MONEY'}
# However, we use tagtog's useful API to generalize the mapping:
map_ids_to_names = get_tagtog_anntasks_json_map()
# we just invert the dictionary
map_names_to_ids = {name: class_id for class_id, name in map_ids_to_names.items()}
print(f"The tagtog project has these annotation labels: {map_ids_to_names}")

def get_class_id(label) -> Optional[str]:
  """Translates the predicted label id into the tagtog entity class id"""
  try:
    return map_names_to_ids[label]
  except KeyError as name:
    print(
        f"ERROR. You must add the label {name} to your tagtog project settings ({MY_PROJECT_URL}/-settings) & then restart flask")
    return None


# -----------------------------------------------------------------------------

# Initialize flair models
tagger = SequenceTagger.load(tagger_name)
classifier = TextClassifier.load(classifier_name)
sent_splitter = SegtokSentenceSplitter()

# -----------------------------------------------------------------------------

app = Flask(__name__)
# Handle any POST request coming to the app root path

# -----------------------------------------------------------------------------


def add_vote_for_flair_label(all_label_votes: Dict[str, Dict[str, List[float]]], label_name: str, label: Label) -> Dict[str, Dict[str, List[float]]]:
  votes = all_label_votes[label_name]

  scores = votes.get(label.value, [0, 0])
  scores[0] += label.score
  scores[1] += 1
  votes[label.value] = scores

  all_label_votes[label_name] = votes


def decide_votes_winners(all_label_votes: Dict[str, Dict[str, List[float]]]) -> Dict[str, Tuple[str, float]]:
  ret = {}
  for label_name in all_label_votes:
    scores = all_label_votes[label_name]
    winner = max(scores, key=lambda tuple: tuple[0])
    winner_scores = scores[winner]
    winner_prob = winner_scores[0] / winner_scores[1]
    ret[label_name] = (winner, winner_prob)

  return ret


def mk_entity(e_id: str, part_id: str, text: str, start: int, prob: float, who: str = TAGTOG_TAGGER_WHO, state: str = "pre-added") -> Dict[str, Any]:
  ret = {
    # entity type id
    'classId': e_id,
    'part': part_id,
    # entity offset
    'offsets': [{'start': start, 'text': text}],
    # entity confidence object (annotation status, who created it and probabilty)
    'confidence': {'state': state, 'who': [who], 'prob': prob},
    # no entity labels (fields)
    'fields': {},
    # this is related to the kb_id
    # (knowledge base ID) field from the Span spaCy object
    'normalizations': {}}

  # print(ret)
  return ret


def mk_doclabel(m_id: str, value: Union[str, bool], prob: float, who: str = TAGTOG_CLASSIFIER_WHO, state: str = "pre-added") -> Dict[str, Any]:
  ret = {
      # document label id
      m_id: {
        'value': value,
        # entity confidence object (annotation status, who created it and probabilty)
        'confidence': {'state': state, 'who': [who], 'prob': prob}
      }
  }

  # print(ret)
  return ret


def mk_empty_annjson() -> Dict[str, Any]:
  return {
    # Set the document as not confirmed; an annotator will later manually confirm whether the annotations are correct
    'anncomplete': False,
    'sources': [],
    'metas': {},
    'relations': [],
    'entities': []
  }


# Spec: https://docs.tagtog.net/anndoc.html#ann-json
def mk_annjson(entities: List[Dict[str, Any]] = None, doclabels: Dict[str, Any] = None, in_ann_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  annjson = mk_empty_annjson() if in_ann_json is None else in_ann_json

  # We merge the new annotations with existing ones, if any
  if doclabels is not None:
    annjson['metas'] = {**annjson['metas'], **doclabels}
  if entities is not None:
    annjson['entities'] += entities

  return annjson


def _has_part_id(elem):
  return elem.has_attr("id")


def gen_parts_generator_over_plain_html_file(plain_html_filename):
  with open(plain_html_filename, "r") as f:
    plain_html_raw = f.read()
    return gen_parts_generator_over_plain_html(plain_html_raw)


def gen_parts_generator_over_plain_html(plain_html_raw):
  plain_html_soup = BeautifulSoup(plain_html_raw, "html.parser")

  for partElem in plain_html_soup.body.find_all(_has_part_id):
    yield partElem


def annotate(plain_html, in_ann_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  entities = []
  all_label_votes = {DOC_LABEL_SENTIMENT_NAME: {}}

  for part in gen_parts_generator_over_plain_html(plain_html):
    part_id = part.get('id')
    text = part.text

    sentences = sent_splitter.split(text)

    tagger.predict(sentences)
    classifier.predict(sentences)

    # iterate through sentences and parse out entities
    for sentence in sentences:
        for label in sentence.get_labels():
          add_vote_for_flair_label(all_label_votes, DOC_LABEL_SENTIMENT_NAME, label)

        for entity in sentence.get_spans(label_type='ner'):
          for entity_class in entity.labels:
            e_id = get_class_id(entity_class.value)
            if e_id:
              # Adjust the entity start offset relative to the original text
              #   NOTE: flair keeps the offset relative to the original text in the sentences, but unfortunately loses the information in the predicted entities/spans
              adjusted_start = entity.start_pos + sentence.start_pos
              entity = mk_entity(e_id=e_id, part_id=part_id,
                                 text=entity.text, start=adjusted_start, prob=entity_class.score)
              entities.append(entity)

  print(all_label_votes)
  winners = decide_votes_winners(all_label_votes)
  print(winners)

  doc_labels = {}
  for label_name, (value, prob) in winners.items():
    m_id = get_class_id(label_name)
    if m_id:
      doc_label = mk_doclabel(m_id, value, prob)
      doc_labels = {**doc_labels, **doc_label}

  print(doc_labels)

  ret = mk_annjson(entities=entities, doclabels=doc_labels,
                   in_ann_json=in_ann_json)
  # print(ret)
  return ret

# -----------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def ping():
  return "Yes, I'm here!"


@app.route('/', methods=['POST'])
def respond():
  print(f"Received a webhook signal: {request.json}")

  docid = request.json.get('tagtogID')

  # Parameters for the GET API call to get a document
  # (see https://docs.tagtog.net/API_documents_v1.html#examples-get-the-original-document-by-document-id)
  get_params = {**default_API_params, **{'ids': docid}}
  get_params_plain_doc = {**get_params, **{'output': 'plain.html'}}
  get_params_ann_doc = {**get_params, **{'output': 'ann.json'}}

  # Request plain.html file from tagtog, which contains all the document's content
  get_plain_response = requests.get(
      tagtog_docs_API_endpoint, params=get_params_plain_doc, auth=auth, verify=VERIFY_SSL_CERT)
  plain_html = get_plain_response.content

  # Request ann.json file from tagtog, which might already contain pre-annotations
  get_ann_response = requests.get(
      tagtog_docs_API_endpoint, params=get_params_ann_doc, auth=auth, verify=VERIFY_SSL_CERT)
  try:
    in_ann_json = get_ann_response.json()
  except:
    in_ann_json = None

  is_new_doc = request.headers.get('X-tagtog-onPushSave-status') == 'created'

  if is_new_doc:
    # annotate

    out_ann_json = annotate(plain_html, in_ann_json)

    # Pre-annotated document composed of the content and the annotations
    files = [(docid + '.plain.html', plain_html),
             (docid + '.ann.json', json.dumps(out_ann_json))]

    # Upload to tagtog our predicted annotations
    post_response = requests.post(tagtog_docs_API_endpoint, params=post_params_doc, auth=auth, files=files, verify=VERIFY_SSL_CERT)
    print(post_response.text)

  return '', 204

# -----------------------------------------------------------------------------

if __name__ == "__main__":
  app.run(host='0.0.0.0')
