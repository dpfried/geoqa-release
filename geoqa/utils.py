import io
import numpy as np
import sexpdata
from collections import namedtuple

_Stack = namedtuple('_Stack', ['top', 'rest', 'size'])
class Stack(_Stack):
    """Functional stack which allows sharing memory in a beam search"""
    @staticmethod
    def empty():
        return Stack(None, None, 0)

    @staticmethod
    def make(top, rest):
        if top is None:
            return Stack.empty()
        if rest is None:
            return Stack(top, None, 1)
        else:
            return Stack(top, rest, rest.size + 1)

    def __contains__(self, item):
        return item in self.tolist()

    def __iter__(self):
        items = self.tolist()
        return items.__iter__()

    def __repr__(self):
        return '{}'.format(self.tolist())

    def tolist(self, limit=None):
        acc = []
        stack = self
        while (limit is None or limit > 0) and stack.top is not None:
            acc.append(stack.top)
            stack = stack.rest
            if limit is not None:
                limit -= 1
        return list(reversed(acc))

    def append(self, item):
        return Stack.make(item, self)

class Index(object):
  def __init__(self):
    self.items = []
    self.indices = {}
    self.frozen = False

  def __contains__(self, item):
    return item in self.indices

  def index(self, item):
    if item not in self.indices:
      assert not self.frozen
      index = len(self.indices)
      self.indices[item] = index
      self.items.append(item)
    return self.indices[item]

  def get(self, index):
    return self.items[index]

  def size(self):
    return len(self.items)


def get_word_vectors(filename, filter_tokens=None, lowercase=True, max_dim=None):
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if lowercase:
            word = word.lower()
        if (filter_tokens is None) or (word in filter_tokens):
            if word not in data:
                data[word] = np.array([float(tok) for tok in tokens[1:]])
                if max_dim is not None:
                    data[word] = data[word][:max_dim]
    return data

def parse_tree(p):
    if "'" in p:
        p = "none"
    parsed = sexpdata.loads(p)
    extracted = extract_parse(parsed)
    return extracted

def extract_parse(p):
    if isinstance(p, sexpdata.Symbol):
        return p.value()
    elif isinstance(p, int):
        return str(p)
    elif isinstance(p, bool):
        return str(p).lower()
    elif isinstance(p, float):
        return str(p).lower()
    return tuple(extract_parse(q) for q in p)

def logical_form_to_str(logical_form):
  return sexpdata.dumps(logical_form, str_as='symbol')
