from . import util
from keras import callbacks as kc
from keras import models as km
from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from keras.models import Model
from sklearn.metrics import f1_score
import numpy as np
import os
import random
import tensorflow as tf

class _Model:

    # Paths to store and load the model architecture and weights
    MODEL_DIR = 'repo'
    MODEL_CONFIG = 'model.json'
    MODEL_WEIGHTS = 'model.weights.h5'

    def __init__(self, labels, filename):
        # Initialize with provided labels and filename
        self.labels = labels
        self.num_classes = len(labels)
        self.model_name = 'model_' + filename  # Unique model identifier
        self.path = os.path.join(self.MODEL_DIR, 'models', self.model_name)
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
        self.model.summary()
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
        - x: Input data, must be a list/array-like object or a resolved tensor.
        - y: Labels, must be a list/array-like object or a TensorFlow tensor.
        - epochs: Number of training epochs.
        """
        try:
            # Ensure x and y are not symbolic tensors
            if hasattr(x, '_keras_history') or hasattr(x, 'op'):
                raise ValueError(
                    "Input `x` is a symbolic tensor (KerasTensor). Provide actual data as NumPy arrays or eager tensors."
                )

            # Convert x to NumPy array if needed
            if isinstance(x, list):
                x = np.array([self.to_x(trial) for trial in x])  # Convert list of trials to NumPy array
            elif tf.is_tensor(x):
                x = x.numpy()  # Convert TensorFlow tensor to NumPy array
            elif not isinstance(x, np.ndarray):
                x = np.array(x)  # Try converting any other type to NumPy array

            # Convert y to NumPy array if needed
            if isinstance(y, list):
                y = np.array([self.to_y(trial) for trial in y])  # Convert list of labels to NumPy array
            elif tf.is_tensor(y):
                y = y.numpy()  # Convert TensorFlow tensor to NumPy array
            elif not isinstance(y, np.ndarray):
                y = np.array(y)  # Try converting any other type to NumPy array

            # Add a channel dimension if x has 3 dimensions
            if x.ndim == 3:
                x = np.expand_dims(x, axis=-1)  # Add channel dimension

            # Check shapes and ensure compatibility
            if x.ndim != 4:
                raise ValueError(f"Input `x` must have 4 dimensions (batch_size, height, width, channels), but got {x.shape}.")
            if y.ndim != 2:
                raise ValueError(f"Labels `y` must have 2 dimensions (batch_size, num_classes), but got {y.shape}.")
        except Exception as e:
            print(f"Error during data preparation: {e}")
            raise e

        # Split data into training and validation sets
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)

        best = None

        def check_best(epoch, logs):
            nonlocal best
            if (best is None
                    or logs['val_accuracy'] > best['val_accuracy']
                    or (logs['val_accuracy'] == best['val_accuracy'] and logs['accuracy'] > best['accuracy'])):
                best = dict(logs)
                best['epoch'] = epoch + 1
                best['file'] = f"_model_{best['epoch']:03d}_{best['accuracy']:.4f}_{best['val_accuracy']:.4f}.weights.h5"
                print('\nSaving best model so far: ' + best['file'])
                self.model.save_weights(os.path.join(self.path, best['file']))

        callbacks = [
            kc.LambdaCallback(on_epoch_end=check_best),
        ]

        # Fit the model
        self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=5,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, y_val),  # Provide validation data directly
        )

        if best:
            print('Restoring best model: ' + best['file'])
            self.model.load_weights(os.path.join(self.path, best['file']))

        self.save()

        return {
            'val_loss': best['val_loss'],
            'val_accuracy': best['val_accuracy'],
        }

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
            correct = 0
            total = 0
            for trial in trials:
                y = self.predict(trial)
                ys = '[' + ' '.join([f'{yy:.3f}' for yy in y]) + ']'
                total += 1
                label = self.labels[np.argmax(y)]
                if label == trial.label:
                    correct += 1

    def evaluate_crossfold(self, dataset, num_folds=10, epochs=25):
        """
        Evaluate the model using K-Fold cross-validation on the entire dataset.
        Args:
        - dataset: List of trials or tuple (x, y) for evaluation.
        - num_folds: Number of folds for K-Fold cross-validation.
        - epochs: Number of epochs for training in each fold.
        """
        def prepare_data(data):
            if hasattr(data, 'trials'):
                trials = data.trials
                return self.to_x_y(trials)
            elif isinstance(data, tuple) and len(data) == 2:
                return data
            elif isinstance(data, list):  # Handle lists of trials
                return self.to_x_y(data)
            else:
                raise TypeError("Input data must be a dataset object with 'trials', a tuple (x, y), or a list of trials.")

        def k_fold_evaluation(data_points, labels, num_folds, epochs):
            # Ensure alignment of data and labels
            if len(data_points) != len(labels):
                min_length = min(len(data_points), len(labels))
                data_points, labels = data_points[:min_length], labels[:min_length]

            # Shuffle data
            combined = list(zip(data_points, labels))
            np.random.shuffle(combined)
            data_points, labels = zip(*combined)
            data_points, labels = np.array(data_points), np.array(labels)

            # K-Fold Cross-Validation
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
                self._train(train_data, train_labels, epochs=epochs)

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

        # Prepare the entire dataset
        data_points, labels = prepare_data(dataset)

        # Perform K-Fold evaluation
        mean_accuracy, mean_f1 = k_fold_evaluation(data_points, labels, num_folds, epochs)

        return {
            "mean_accuracy": mean_accuracy,
            "mean_f1_score": mean_f1,
        }
    
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

class MobileNetModel(_Model):
    def train(self, trials, epochs=50):
        # Shuffle trials to ensure randomized input data for training.
        random.shuffle(trials)

        # Convert trials to input features (x) and labels (y).
        x, y = self.to_x_y(trials)
        
        # Add channel dimension for compatibility with Conv2D layers.
        # Depthwise separable convolutions in MobileNetV1 operate on 4D input tensors (batch_size, height, width, channels).
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)

        # Ensure labels (y) are converted to a NumPy array for processing.
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Define input shape.
        # MobileNetV1 utilizes a structured input format suitable for depthwise separable convolutions.
        input_shape = (x.shape[1], 6, 1)  # 10 frames, 6 features, 1 channel
        
        # Build MobileNet model.
        inputs = Input(shape=input_shape)

        # Standard convolution layer to extract low-level features.
        # MobileNetV1 begins with a standard convolution layer, as described in Section 3.2 of the paper [9].
        x_in = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x_in = BatchNormalization()(x_in)
        x_in = ReLU()(x_in)

        # Depthwise separable convolution (depthwise + pointwise convolution).
        # MobileNetV1 replaces standard convolutions with depthwise separable convolutions for efficiency, as described in Section 3.1 [9].
        x_in = DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False)(x_in)
        x_in = BatchNormalization()(x_in)
        x_in = ReLU()(x_in)

        # Pointwise convolution to combine features from the depthwise layer.
        # This step reduces dimensionality while combining features [9].
        x_in = Conv2D(64, kernel_size=(1, 1), use_bias=False)(x_in)
        x_in = BatchNormalization()(x_in)
        x_in = ReLU()(x_in)

        # Additional depthwise separable convolution block.
        # MobileNetV1 repeats this structure to build deeper feature representations, as defined in Table 1 [9].
        x_in = DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False)(x_in)
        x_in = BatchNormalization()(x_in)
        x_in = ReLU()(x_in)
        x_in = Conv2D(128, kernel_size=(1, 1), use_bias=False)(x_in)
        x_in = BatchNormalization()(x_in)
        x_in = ReLU()(x_in)

        # Global Average Pooling layer to aggregate spatial information.
        # This is a lightweight approach to summarize spatial features, common in MobileNet architectures [9].
        x_in = GlobalAveragePooling2D()(x_in)

        # Fully connected (Dense) layer for classification.
        # The output layer uses softmax for multiclass classification tasks.
        outputs = Dense(len(self.labels), activation='softmax')(x_in)

        # Compile the model with Adam optimizer and categorical cross-entropy loss.
        # MobileNetV1 models are trained using standard optimizers and loss functions suited for classification [9].
        self.model = Model(inputs, outputs)
        self.model.summary()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model on the preprocessed data.
        # Training involves iterating over the input data for the specified number of epochs.
        self._train(x, y, epochs)
