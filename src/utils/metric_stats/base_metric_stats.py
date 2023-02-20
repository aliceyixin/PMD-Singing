import torch
import numpy as np

class BaseMetricStats:
    def __init__(self, metric_fn=None):
        self.clear()
        self.metric_fn = metric_fn

    def clear(self):
        self.metric_keys = []
        self.ids = []
        self.scores_list = []

    def append(self, ids, **kwargs):
        if self.metric_fn is None:
            raise ValueError('No metric_fn has been provided')
        self.ids.extend(ids)  # save ID
        self.scores_list.extend(self.metric_fn(**kwargs))  # save metrics
        if len(self.metric_keys) == 0:  # save metric keys
            self.metric_keys = list(self.scores_list[0].keys())

    def summarize(self, field=None):
        if len(self.metric_keys) == 0:
            raise ValueError('No metrics saved yet')

        mean_scores = {}
        for key in self.metric_keys:  # calculate average scores
            mean_scores[key] = [scores[key] for scores in self.scores_list]
            mean_scores[key] = torch.tensor(np.mean(mean_scores[key]))

        if field is None:
            return mean_scores
        else:
            return mean_scores[field]

    def write_stats(self, f):
        scores = self.summarize()
        f.write('\t'.join([str(value) for key, value in scores.items()]) + '\n')