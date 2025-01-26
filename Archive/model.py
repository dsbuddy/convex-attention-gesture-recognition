import os
import glob
import random
from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from keras import callbacks as kc
from keras import layers as kl
from keras import models as km

import util

class _Model:
    MODEL_DIR = './'
    MODEL_CONFIG = 'model.json'
    MODEL_WEIGHTS = 'model.hdf5'

    def __init__(self, labels, filename):
        self.labels = labels
        name = 'model_' + filename
        self.path = os.path.join(self.MODEL_DIR, 'models', name)
        os.makedirs(self.path, exist_ok=True)
        self.model = None

    def __str__(self):
        if self.model:
            lines = []
            self.model.summary(print_fn=lambda line: lines.append(line))
            return '\n'.join(lines)
        else:
            return ''

    def load(self):
        self.model = km.model_from_json(util.load_text(os.path.join(self.path, self.MODEL_CONFIG)))
        self.model.load_weights(os.path.join(self.path, self.MODEL_WEIGHTS))
        # self.model.summary()
        return self

    def save(self):
        util.save_text(os.path.join(self.path, self.MODEL_CONFIG),self.model.to_json())
        self.model.save_weights(os.path.join(self.path, self.MODEL_WEIGHTS))
        return self

    def to_x(self, trial):
        return np.array(trial.data)

    def to_y(self, trial):
        return util.one_hot(self.labels.index(trial.label), len(self.labels))

    def to_x_y(self, trials):
        return(
            np.array([self.to_x(trial) for trial in trials]),
            np.array([self.to_y(trial) for trial in trials]),
        )

    def _train(self, x, y, epochs):

        best = None

        def check_best(epoch, logs):
            nonlocal best
            if (best is None
                or logs['val_acc'] > best['val_acc']
                    or (logs['val_acc'] == best['val_acc'] and logs['acc'] > best['acc'])):
                best = dict(logs)
                best['epoch'] = epoch + 1
                best['file'] = f"_model_{best['epoch']:03d}_{best['acc']:.4f}_{best['val_acc']:.4f}.hdf5"
                print('\nSaving best model so far: ' + best['file'])
                self.model.save_weights(os.path.join(self.path, best['file']))

        callbacks = [kc.LambdaCallback(on_epoch_end=check_best),]

        self.model.fit(
            x=x,
            y=y,
            batch_size=5,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=.3,
        )

        if best:
            print('Restoring best model: ' + best['file'])
            self.model.load_weights(os.path.join(self.path, best['file']))

        self.save()

        return {
            'val_loss': best['val_loss'],
            'val_acc': best['val_acc'],
        }
    
    def cleanUpModels(self):
    	files = glob.glob(self.path+"/_model*")
    	for f in files:
    		os.remove(f)
    

    def predict(self, trial):
        x = np.array([self.to_x(trial)])
        y = self.model.predict(x, verbose=0)[0]
        return y

    def classify(self, trial):
        y = self.predict(trial)
        index = np.argmax(y)
        return self.labels[index]

    def evaluate(self, training_set, test_set):
        for trials in [training_set, test_set]:
            print()
            correct = 0
            total = 0
            for trial in trials:
                y = self.predict(trial)
                ys = '[' + ' '.join([f'{yy:.3f}' for yy in y]) + ']'
                print(f'{trial}\t{ys}')
                total += 1
                label = self.labels[np.argmax(y)]
                if label == trial.label:
                    correct += 1
            print(f'\nCorrect: {correct} / {total} = {correct/total:.3f}\n')


class PadModel(_Model):
    def train(self, trials, epochs=200):
        random.shuffle(trials)
        x, y = self.to_x_y(trials)
        
        numFrame = x[0].shape[0]
        numChannel = x[0].shape[1]

        inputs = kl.Input(shape=x[0].shape)
        outputs = inputs
        for layer in [
            kl.Reshape((1, numFrame, numChannel)),
            kl.Conv2D(filters=16, kernel_size=(1, 3), activation='relu'),
            kl.MaxPooling2D(pool_size=(1, 2)),
            kl.Flatten(),
            kl.Dense(16, activation='relu'),
            kl.Dense(len(self.labels), activation='softmax'),
        ]:
            outputs = layer(outputs)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])

        self._train(x, y, epochs)
