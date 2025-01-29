import os
import random
from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from keras import callbacks as kc
from keras import layers as kl
from keras import models as km

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import numpy as np

import util
from data import *


class _Model:

    MODEL_DIR = 'repo'
    MODEL_CONFIG = 'model.json'
    MODEL_WEIGHTS = 'model.weights.h5'

    def __init__(self, labels, filename):
        self.labels = labels
        name = 'model_' + filename #datetime.now().strftime('%Y%m%d_%H%M%S')
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
        self.model = km.model_from_json(util.load_text(
            os.path.join(self.path, self.MODEL_CONFIG)))
        self.model.load_weights(os.path.join(self.path, self.MODEL_WEIGHTS))
        # self.model.summary()
        return self

    def save(self):
        util.save_text(os.path.join(self.path, self.MODEL_CONFIG),
                       self.model.to_json())
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
        """
        Train the model with the given data and labels.

        Args:
        - x: Input data, must be a list or array-like object.
        - y: Labels, must be a list or array-like object.
        - epochs: Number of training epochs.
        """
        # Convert x and y to NumPy arrays
        # Assuming 'Trial' represents a custom data type, use a fallback for undefined 'Trial'
        try:
            x = np.array([self.to_x(trial) if isinstance(trial, Trial) else trial for trial in x])
            y = np.array([self.to_y(trial) if isinstance(trial, Trial) else trial for trial in y])
        except NameError:
            print("Warning: 'Trial' is not defined. Using data directly.")
            x = np.array([self.to_x(trial) if hasattr(trial, 'data') else trial for trial in x])
            y = np.array([self.to_y(trial) if hasattr(trial, 'label') else trial for trial in y])

        best = None

        def check_best(epoch, logs):
            nonlocal best
            if (best is None
                    or logs['val_acc'] > best['val_acc']
                    or (logs['val_acc'] == best['val_acc'] and logs['acc'] > best['acc'])):
                best = dict(logs)
                best['epoch'] = epoch + 1
                best['file'] = f"_model_{best['epoch']:03d}_{best['acc']:.4f}_{best['val_acc']:.4f}.weights.h5"
                print('\nSaving best model so far: ' + best['file'])
                self.model.save_weights(os.path.join(self.path, best['file']))

        callbacks = [
            kc.LambdaCallback(on_epoch_end=check_best),
        ]

        # Fit the model
        self.model.fit(
            x=x,
            y=y,
            batch_size=5,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.3,
        )

        if best:
            print('Restoring best model: ' + best['file'])
            self.model.load_weights(os.path.join(self.path, best['file']))

        self.save()

        return {
            'val_loss': best['val_loss'],
            'val_acc': best['val_acc'],
        }


    # def _train(self, x, y, epochs):

    #     best = None

    #     def check_best(epoch, logs):
    #         nonlocal best
    #         if (best is None
    #             or logs['val_acc'] > best['val_acc']
    #                 or (logs['val_acc'] == best['val_acc'] and logs['acc'] > best['acc'])):
    #             best = dict(logs)
    #             best['epoch'] = epoch + 1
    #             best['file'] = f"_model_{best['epoch']:03d}_{best['acc']:.4f}_{best['val_acc']:.4f}.weights.h5"
    #             print('\nSaving best model so far: ' + best['file'])
    #             self.model.save_weights(os.path.join(self.path, best['file']))

    #     callbacks = [
    #         kc.LambdaCallback(on_epoch_end=check_best),
    #     ]

    #     self.model.fit(
    #         x=x,
    #         y=y,
    #         batch_size=5,
    #         epochs=epochs,
    #         callbacks=callbacks,
    #         validation_split=.3,
    #     )

    #     if best:
    #         print('Restoring best model: ' + best['file'])
    #         self.model.load_weights(os.path.join(self.path, best['file']))

    #     self.save()

    #     return {
    #         'val_loss': best['val_loss'],
    #         'val_acc': best['val_acc'],
    #     }

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

    def align_data_and_labels(self, data, labels):
        if len(data) > len(labels):
            data = data[:len(labels)]  # Trim data to match labels
        elif len(data) < len(labels):
            labels = labels[:len(data)]  # Trim labels to match data
        return np.array(data), np.array(labels)

    def evaluate_crossfold(self, data, num_folds=10):
        """
        Evaluate the model using manually implemented K-Fold cross-validation with accuracy and macro F1-score.

        Args:
        - data: A `Data` object containing trials with input data and labels.
        - num_folds: Number of folds for K-Fold cross-validation.
        """
        # Extract trials from the Data object
        if not hasattr(data, 'trials'):
            raise TypeError("The provided 'data' object must have a 'trials' attribute containing the dataset.")

        trials = data.trials  # Access the trials list
        if not isinstance(trials, list):
            raise TypeError("'trials' attribute must be a list of Trial objects.")

        # Extract data points and labels from the trials
        data_points, labels = self.to_x_y(trials)

        # Ensure data and labels are aligned
        if len(data_points) != len(labels):
            min_length = min(len(data_points), len(labels))
            print(f"Warning: Data and labels are not aligned. Trimming to {min_length} samples.")
            data_points = data_points[:min_length]
            labels = labels[:min_length]

        # Shuffle data and labels together
        combined = list(zip(data_points, labels))
        np.random.shuffle(combined)
        data_points, labels = zip(*combined)
        data_points = np.array(data_points)
        labels = np.array(labels)

        # Split data into folds
        fold_size = len(data_points) // num_folds
        all_accuracies = []
        all_f1_scores = []
        all_scores = []

        for fold in range(num_folds):
            print(f"\nFold {fold + 1}/{num_folds}")

            # Create train and test splits manually
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold != num_folds - 1 else len(data_points)

            test_data = data_points[test_start:test_end]
            test_labels = labels[test_start:test_end]

            train_data = np.concatenate((data_points[:test_start], data_points[test_end:]), axis=0)
            train_labels = np.concatenate((labels[:test_start], labels[test_end:]), axis=0)

            # Train the model on the training fold
            self._train(train_data, train_labels, epochs=25)

            # Predict on the test fold
            predictions = np.argmax(self.model.predict(test_data), axis=1)
            true_labels = np.argmax(test_labels, axis=1)

            # Calculate metrics
            accuracy = np.mean(predictions == true_labels)
            f1 = f1_score(true_labels, predictions, average='macro')

            all_accuracies.append(accuracy)
            all_f1_scores.append(f1)

            all_scores.append(f"Accuracy: {accuracy:.4f}, Macro F1 Score: {f1:.4f}")

        # Report mean metrics over all folds
        mean_accuracy = np.mean(all_accuracies)
        mean_f1 = np.mean(all_f1_scores)
        for i, score in enumerate(all_scores):
            print(f"Fold {i + 1} {score}")

        print(f"\nMean Accuracy over {num_folds} folds: {mean_accuracy:.4f}")
        print(f"Mean Macro F1 Score over {num_folds} folds: {mean_f1:.4f}")

        return mean_accuracy, mean_f1




    # def evaluate_rigorous(self, data, labels, num_folds=10):
    #     """
    #     Evaluate the model using manually implemented K-Fold cross-validation with accuracy and macro F1-score.

    #     Args:
    #     - data: List or numpy array of input data.
    #     - labels: List or numpy array of true labels.
    #     - num_folds: Number of folds for K-Fold cross-validation.
    #     """
    #     # Ensure data and labels are aligned
    #     if len(data) != len(labels):
    #         min_length = min(len(data), len(labels))
    #         print(len(data))
    #         print(len(labels))
    #         print(f"Warning: Data and labels are not aligned. Trimming to {min_length} samples.")
    #         exit()
    #         data = data[:min_length]
    #         labels = labels[:min_length]

    #     # Shuffle data and labels together
    #     combined = list(zip(data, labels))
    #     np.random.shuffle(combined)
    #     data, labels = zip(*combined)
    #     data = np.array(data)
    #     labels = np.array(labels)

    #     # Split data into folds
    #     fold_size = len(data) // num_folds
    #     all_accuracies = []
    #     all_f1_scores = []

    #     for fold in range(num_folds):
    #         print(f"\nFold {fold + 1}/{num_folds}")

    #         # Create train and test splits manually
    #         test_start = fold * fold_size
    #         test_end = test_start + fold_size if fold != num_folds - 1 else len(data)

    #         test_data = data[test_start:test_end]
    #         test_labels = labels[test_start:test_end]

    #         train_data = np.concatenate((data[:test_start], data[test_end:]), axis=0)
    #         train_labels = np.concatenate((labels[:test_start], labels[test_end:]), axis=0)

    #         # Train the model on the training fold
    #         self._train(train_data, train_labels, epochs=200)

    #         # Predict on the test fold using `predict` for consistency
    #         predictions = [np.argmax(self.predict(trial)) for trial in test_data]
    #         true_labels = [np.argmax(label) for label in test_labels]

    #         # Calculate metrics
    #         accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    #         f1 = f1_score(true_labels, predictions, average='macro')

    #         all_accuracies.append(accuracy)
    #         all_f1_scores.append(f1)

    #         print(f"Accuracy: {accuracy:.4f}, Macro F1 Score: {f1:.4f}")

    #     # Report mean metrics over all folds
    #     mean_accuracy = np.mean(all_accuracies)
    #     mean_f1 = np.mean(all_f1_scores)

    #     print(f"\nMean Accuracy over {num_folds} folds: {mean_accuracy:.4f}")
    #     print(f"Mean Macro F1 Score over {num_folds} folds: {mean_f1:.4f}")

    #     return mean_accuracy, mean_f1

    def save_tflite(self):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        # the quantization/optimization below breaks the model on arduino!
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite = converter.convert()

        path = os.path.join(self.path, 'model.tflite')
        with open(path, 'wb') as f:
            f.write(tflite)

        path = os.path.join(self.path, 'flatbuffer.cpp')
        with open(path, 'w') as f:
            f.write('#include "flatbuffer.h"\n\n')
            f.write(f'int FlatBuffer::labelsLength = {len(self.labels)};\n\n')
            f.write('char *FlatBuffer::labels[] = {')
            f.write(', '.join([f'"{l}"' for l in self.labels]))
            f.write('};\n\n')
            f.write(f'int FlatBuffer::bytesLength = {len(tflite)};\n\n')
            f.write('unsigned char FlatBuffer::bytes[] = {\n   ')
            for i, val in enumerate(tflite):
                f.write(f' 0x{val:02x}')
                if (i + 1) < len(tflite):
                    f.write(',')
                    if (i + 1) % 12 == 0:
                        f.write('\n   ')
                else:
                    f.write('\n')
            f.write('};\n')


class PadModel(_Model):

    def train(self, trials, epochs=50):

        random.shuffle(trials)
        x, y = self.to_x_y(trials)
        
        numFrame = x[0].shape[0]
        numChannel = x[0].shape[1]

        inputs = kl.Input(shape=x[0].shape)
        outputs = inputs
        for layer in [

            kl.Reshape((1, numFrame, numChannel)),

            # kl.Conv2D(filters=32, kernel_size=(1, 5), activation='relu'),
            # kl.MaxPooling2D(pool_size=(1, 2)),
            # kl.Conv2D(filters=32, kernel_size=(1, 5), activation='relu'),
            # kl.MaxPooling2D(pool_size=(1, 2)),
            # kl.Flatten(),
            # kl.Dense(32, activation='relu'),
            # kl.Dense(16, activation='relu'),
            # kl.Dense(len(self.labels), activation='softmax'),

            kl.Conv2D(filters=16, kernel_size=(1, 3), activation='relu'),
            kl.MaxPooling2D(pool_size=(1, 2)),
            # kl.Conv2D(filters=16, kernel_size=(1, 3), activation='relu'),
            # kl.MaxPooling2D(pool_size=(1, 2)),
            kl.Flatten(),
            kl.Dense(16, activation='relu'),
            # kl.Dense(8, activation='relu'),
            kl.Dense(len(self.labels), activation='softmax'),
        ]:
            outputs = layer(outputs)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['acc'])

        self._train(x, y, epochs)



class PadAttentionModel(_Model):

    def train(self, trials, epochs=50):
        random.shuffle(trials)
        x, y = self.to_x_y(trials)
        
        numFrame = x[0].shape[0]
        numChannel = x[0].shape[1]

        inputs = kl.Input(shape=x[0].shape)
        reshaped = kl.Reshape((1, numFrame, numChannel))(inputs)

        # Convolutional Layers
        conv = kl.Conv2D(filters=16, kernel_size=(1, 3), activation='relu')(reshaped)
        pooled = kl.MaxPooling2D(pool_size=(1, 2))(conv)

        # Flatten for attention
        flattened = kl.Flatten()(pooled)

        # Add Traditional Attention Mechanism
        query = kl.Dense(16, activation="relu")(flattened)  # Shape: (None, 16)
        key = kl.Dense(16, activation="relu")(flattened)    # Shape: (None, 16)
        value = kl.Dense(16, activation="relu")(flattened)  # Shape: (None, 16)

        # Compute attention scores
        attention_scores = kl.Dot(axes=-1)([query, key])  # Shape: (None, 1)
        attention_scores = kl.Activation("softmax")(attention_scores)  # Normalize scores

        # Expand dimensions to align with value tensor
        attention_scores = kl.Reshape((1, 1))(attention_scores)  # Shape: (None, 1, 1)

        # Expand value tensor for broadcasting
        value = kl.Reshape((1, 16))(value)  # Shape: (None, 1, 16)

        # Compute weighted sum of values
        attention_output = kl.Multiply()([attention_scores, value])  # Shape: (None, 1, 16)
        attention_output = kl.Flatten()(attention_output)  # Shape: (None, 16)

        # Combine attention output with final layers
        combined = kl.Concatenate()([flattened, attention_output])
        dense_1 = kl.Dense(16, activation='relu')(combined)
        outputs = kl.Dense(len(self.labels), activation='softmax')(dense_1)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['acc'])

        self._train(x, y, epochs)




# class ConvexSelfAttention(kl.Layer):
#     def __init__(self, num_heads, embed_dim, **kwargs):
#         super(ConvexSelfAttention, self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#         self.query_dense = kl.Dense(embed_dim)
#         self.key_dense = kl.Dense(embed_dim)
#         self.value_dense = kl.Dense(embed_dim)

#     def call(self, inputs):
#         queries = self.query_dense(inputs)
#         keys = self.key_dense(inputs)
#         values = self.value_dense(inputs)

#         attention_scores = tf.matmul(queries, keys, transpose_b=True)
#         attention_scores /= tf.sqrt(tf.cast(self.embed_dim, tf.float32))
#         attention_weights = tf.nn.softmax(attention_scores, axis=-1)

#         attention_output = tf.matmul(attention_weights, values)
#         return attention_output


# class ConvexMLPMixer(kl.Layer):
#     def __init__(self, num_tokens, num_features, **kwargs):
#         super(ConvexMLPMixer, self).__init__(**kwargs)
#         self.token_mixing = kl.Dense(num_tokens, activation='softmax')
#         self.feature_mixing = kl.Dense(num_features, activation='relu')

#     def call(self, inputs):
#         token_mixed = self.token_mixing(inputs)
#         feature_mixed = self.feature_mixing(token_mixed)
#         return feature_mixed


# class ConvexFourierOperator(kl.Layer):
#     def __init__(self, embed_dim, **kwargs):
#         super(ConvexFourierOperator, self).__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.fourier_dense = kl.Dense(embed_dim)

#     def call(self, inputs):
#         fourier_transformed = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
#         transformed = self.fourier_dense(tf.abs(fourier_transformed))
#         inverse_transformed = tf.signal.ifft2d(tf.cast(transformed, tf.complex64))
#         return tf.abs(inverse_transformed)


# class ConvexAttentionModel(_Model):
#     def train(self, trials, epochs=50):
#         random.shuffle(trials)
#         x, y = self.to_x_y(trials)

#         numFrame = x[0].shape[0]
#         numChannel = x[0].shape[1]

#         inputs = kl.Input(shape=x[0].shape)

#         reshaped = kl.Reshape((numFrame, numChannel))(inputs)
#         attention_output = ConvexSelfAttention(num_heads=4, embed_dim=numChannel)(reshaped)
#         mlp_mixed = ConvexMLPMixer(num_tokens=numFrame, num_features=numChannel)(attention_output)
#         fourier_output = ConvexFourierOperator(embed_dim=numChannel)(mlp_mixed)

#         flattened = kl.Flatten()(fourier_output)
#         dense_1 = kl.Dense(64, activation='relu')(flattened)
#         outputs = kl.Dense(len(self.labels), activation='softmax')(dense_1)

#         self.model = km.Model(inputs=inputs, outputs=outputs)
#         self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#         self._train(x, y, epochs)
