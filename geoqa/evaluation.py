import torch
import numpy as np
from torch.utils.data import DataLoader

from geoqa.utils import logical_form_to_str
from geoqa.executor import execute

def evaluate_predictions(dataset, name, prediction_function,
                         loss_function=None, display_predictions_frequency=None,
                         logical_form_output_file=None, denotation_output_file=None):
  denotation_matches = []

  dataloader = DataLoader(dataset, shuffle=False, batch_size=1, collate_fn=lambda x: x)

  losses = []
  if logical_form_output_file is not None:
    f_lf = open(logical_form_output_file, 'w')
  else:
    f_lf = None
  if denotation_output_file is not None:
    f_d = open(denotation_output_file)
  else:
    f_d = None

  with torch.no_grad():
    for i, batch in enumerate(dataloader):
      assert len(batch) == 1
      instance = batch[0]

      true_denotation = instance['denotation']
      true_logical_form = instance['logical_form']

      pred_logical_form = prediction_function(instance)
      if pred_logical_form is not None:
        pred_denotation = execute(pred_logical_form, instance['world'])
      else:
        pred_denotation = None

      if loss_function is not None:
        losses.append(
          loss_function(instance)
        )

      true_lf_str = logical_form_to_str(true_logical_form)
      pred_lf_str = logical_form_to_str(pred_logical_form)

      denotation_match = (true_denotation == pred_denotation)
      denotation_matches.append(denotation_match)

      if display_predictions_frequency is not None and i % display_predictions_frequency == 0:
        print("{} example {}".format(name, i+1))
        print("{} question: {}".format(name, instance['question']))
        print("{} true LF: {}".format(name, true_lf_str))
        print("{} pred LF: {}".format(name, pred_lf_str))
        print("{} true denotation: {}".format(name, true_denotation))
        print("{} pred denotation: {}".format(name, pred_denotation))
        print("{} denotation match: {}".format(name, denotation_match))
        print()

      if f_lf is not None:
        f_lf.write("{} ||| {}\n".format(true_lf_str, pred_lf_str))
      if f_d is not None:
        f_d.write("{} ||| {}\n".format(true_denotation, pred_denotation))

  if f_lf is not None:
    f_lf.close()
  if f_d is not None:
    f_d.close()

  stats = {
    'denotation_acc': np.mean(denotation_matches),
  }
  if losses:
    stats['loss'] = np.mean(losses)

  if name != '':
    stats = {
      '{}_{}'.format(name, key): value
      for key, value in stats.items()
    }
  return stats
