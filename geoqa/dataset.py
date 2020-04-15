import geoqa
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import namedtuple
import numpy as np

from geoqa.geo import VARS

Token = namedtuple("Token", [
  "word",  # str
  "word_embedding",  # torch vector
  "predicates",  # a list of predicates
  "predicate_embeddings"
])

class GeoDataset(torch.utils.data.Dataset):
  def __init__(self, state_names, word_to_embedding, logical_form_exclude_function=None):
    self.word_to_embedding = word_to_embedding
    self.embedding_dim = next(iter(word_to_embedding.values())).shape[0]

    self._state_name_to_world = {}

    self._questions = []
    self._tokenized_questions = []
    self._answers = []
    self._logical_forms = []
    self._state_names = []
    self._possible_predicates = []

    for state_name in state_names:
      questions, answers, logical_forms = geoqa.geo.read_data(state_name)
      self._state_name_to_world[state_name] = geoqa.geo.read_world(state_name)
      for question, answer, logical_form in zip(questions, answers, logical_forms):
        if logical_form is None:
          continue
        if logical_form_exclude_function is not None and logical_form_exclude_function(logical_form):
          continue

        question, tokenized_question, possible_predicates = self.tokenize_and_link_question(
          question, state_name,
        )
        self._questions.append(question)
        self._tokenized_questions.append(tokenized_question)
        self._possible_predicates.append(possible_predicates)
        self._state_names.append(state_name)
        self._answers.append(answer)
        self._logical_forms.append(logical_form)

    assert len(self._questions) == len(self._tokenized_questions) == len(self._answers) == len(
      self._logical_forms) == len(self._state_names)

  def __len__(self):
    return len(self._questions)

  def tokenize_and_link_question(self, question, state_name):
    question = question.lower()
    question = question.replace('north west', 'northwest')
    question = question.replace('north east', 'northeast')
    question = question.replace('south west', 'southwest')
    question = question.replace('south east', 'southeast')

    tokenized_question = []

    possible_predicates = set()

    for word in question.split():
      predicates = []
      predicate_embeddings = []
      for predicate_to_keywords in [geoqa.geo.REL_KEYWORDS, geoqa.geo.CAT_KEYWORDS,
                                    geoqa.geo.ENTITY_KEYWORDS_BY_STATE[state_name]]:
        for predicate, keywords in predicate_to_keywords.items():
          embeddings = [self.word_to_embedding[keyword] for keyword in keywords]
          embedding = torch.tensor(np.mean(embeddings, axis=0)).float()
          if word in keywords:
            possible_predicates.add(predicate)
            predicates.append(predicate)
            predicate_embeddings.append(embedding)
      tokenized_question.append(
        Token(word, torch.tensor(self.word_to_embedding[word]).float(), predicates, predicate_embeddings)
      )
    return question, tokenized_question, list(sorted(possible_predicates))

  def __getitem__(self, index):
    answer = self._answers[index]
    state = self._state_names[index]
    world = self._state_name_to_world[self._state_names[index]]
    tokenized_question = self._tokenized_questions[index]

    logical_form = self._logical_forms[index]

    # len(question) x embedding_dim
    embedded_words = torch.stack(
      [token.word_embedding for token in tokenized_question],
      dim=0
    )
    assert embedded_words.size() == (len(tokenized_question), self.embedding_dim)

    predicates_at_each_word_position = [token.predicates for token in tokenized_question]
    max_predicates = max(len(preds) for preds in predicates_at_each_word_position)

    embedded_predicates = [
      torch.stack(token.predicate_embeddings, dim=0)
      if len(token.predicate_embeddings) > 0
      else torch.zeros((0, self.embedding_dim))
      for token in tokenized_question
    ]

    for ps, embs in zip(predicates_at_each_word_position, embedded_predicates):
      assert embs.size(0) == len(ps)

    # embedded_predicates = pad_sequence(
    #   embedded_predicates, batch_first=True
    # )
    # assert embedded_predicates.size() == (len(tokenized_question), max_predicates, self.embedding_dim)

    return {
      'question': self._questions[index],
      'words': [token.word for token in tokenized_question],
      'embedded_words': embedded_words,
      'predicates_at_each_word_position': predicates_at_each_word_position,
      'embedded_predicates_at_each_word_position': embedded_predicates,
      'answer': answer,
      'logical_form': logical_form,
      'world': world,
      'possible_predicates': self._possible_predicates[index],
      'denotation': world.get_denotation_from_answer(answer),
    }
