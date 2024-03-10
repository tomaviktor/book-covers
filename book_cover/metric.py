import abc
from evaluate import load
import numpy as np


class Metrics(abc.ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def compute_metrics(self, p):
        pass


class Accuracy(Metrics):
    def __init__(self, **kwargs):
        self.metric = load("accuracy")

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)

        results = self.metric.compute(predictions=predictions, references=labels.flatten())
        return results
