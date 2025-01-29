import json
import matplotlib.pyplot as plt
import os
from datetime import datetime


class Trial:

    def __init__(self, label, data):
        self.label = label
        self.data = data

    @classmethod
    def from_obj(cls, obj):
        return cls(obj['label'], obj['data'])

    def to_obj(self):
        return {
            'label': self.label,
            'data': self.data,
        }

    def __str__(self):
        return self.label

    def plot(self):
        plt.title(self.label)
        plt.ylim(30, 80)
        plt.plot(self.data)
        plt.show()


class Data:

    def __init__(self, filename=None,addDate=False):
        if(filename==None):
           filename = 'data_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + '.json'
        elif(filename!=None and addDate==True):
            filename = 'data_' + datetime.now().strftime('%Y%m%d_%H%M%S') + filename + '.json'
        #else keep the filename as is

        dir = os.path.join('data')
        os.makedirs(dir, exist_ok=True)
        self.path = os.path.join(dir, filename)
        self.trials = []

    def load(self):
        with open(self.path, 'r') as f:
            obj = json.load(f)
            self.trials = [Trial.from_obj(o) for o in obj]
        return self

    def save(self):
        obj = [t.to_obj() for t in self.trials]
        with open(self.path, 'w') as f:
            json.dump(obj, f)

    def add(self, trial):
        self.trials.append(trial)

    def labels(self):
        labels = set()
        for trial in self.trials:
            labels.add(trial.label)
        return sorted(labels)

    def split(self, test=.3):
        labels = self.labels()

        trials_by_label = {label: [] for label in labels}
        for trial in self.trials:
            trials_by_label[trial.label].append(trial)

        training_set = []
        test_set = []
        for label in labels:
            trials = trials_by_label[label]
            split_point = int(len(trials) * (1 - test))
            training_set.extend(trials[:split_point])
            test_set.extend(trials[split_point:])

        return (training_set, test_set)

    def plot(self):
        for trial in self.trials:
            trial.plot()
