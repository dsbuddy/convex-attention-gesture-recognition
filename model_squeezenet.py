import numpy as np
import os
import random
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Concatenate
from keras.models import Model
from sklearn.metrics import f1_score


class SqueezeNetModel:

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

    def fire_module(self, x, squeeze_filters, expand_filters):
        """
        Implements the Fire Module as described in Section 3.2 of the paper.
        A Fire Module reduces the number of input channels (squeeze layer)
        before applying 1x1 and 3x3 convolutions (expand layer) to balance
        model complexity and accuracy.
        """
        # Squeeze layer with 1x1 convolutions to reduce dimensionality
        squeeze = Conv2D(squeeze_filters, (1, 1), activation="relu", padding="same")(x)
        squeeze = BatchNormalization()(squeeze)

        # Expand layer with a mix of 1x1 and 3x3 convolutions
        expand1x1 = Conv2D(expand_filters, (1, 1), activation="relu", padding="same")(squeeze)
        expand3x3 = Conv2D(expand_filters, (3, 3), activation="relu", padding="same")(squeeze)

        # Concatenation of outputs to combine different receptive fields
        output = Concatenate()([expand1x1, expand3x3])
        return output

    def build_model(self):
        """
        Constructs the CNN based on the SqueezeNet architecture (Section 3.3).
        It starts with a convolutional layer, followed by stacked Fire Modules, 
        and ends with global average pooling and a classification layer.
        """
        # Input layer based on the defined input shape
        inputs = Input(shape=self.input_shape)

        # Initial convolutional layer as per SqueezeNet macroarchitecture (Figure 2)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Two Fire Modules for feature extraction
        x = self.fire_module(x, squeeze_filters=16, expand_filters=32)
        x = self.fire_module(x, squeeze_filters=16, expand_filters=32)

        # Global Average Pooling for spatial data compression
        x = GlobalAveragePooling2D()(x)

        # Final classification layer with softmax activation for output probabilities
        outputs = Dense(self.num_classes, activation="softmax")(x)

        # Compile the model with categorical crossentropy loss and Adam optimizer
        model = Model(inputs, outputs)
        model.summary()  # Summarize the architecture
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    def to_x_y(self, data):
        """
        Convert trials into input (x) and labels (y).
        Args:
        - data: List of Trial objects or a dataset object with a 'trials' attribute.
        Returns:
        - x: Numpy array of inputs.
        - y: Numpy array of one-hot-encoded labels.
        """
        # Handle both list and dataset object formats
        if hasattr(data, 'trials'):  # If it's a Data object
            trials = data.trials
        elif isinstance(data, list):  # If it's already a list of trials
            trials = data
        else:
            raise TypeError("Input data must be a list of trials or an object with a 'trials' attribute.")

        x = np.array([trial.data for trial in trials])
        y = np.array([self.labels.index(trial.label) for trial in trials])
        y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        return x, y

    def train(self, trials, epochs=50):
        """
        Train the SqueezeNet model.
        Args:
        - trials: List of Trial objects or tuples (x, y).
        - epochs: Number of training epochs.
        """
        # Check if trials are tuples (x, y) or Trial objects
        if isinstance(trials[0], tuple):  # Handle (x, y) tuples
            x_train, y_train = zip(*trials)
            x_train = np.array(x_train)
            y_train = np.array(y_train)
        else:  # Handle Trial objects
            x_train, y_train = self.to_x_y(trials)

        # Ensure correct input shape
        x_train = np.expand_dims(x_train, axis=-1)
        # print(x_train.shape)
        # exit()
        self.input_shape = x_train.shape[1:]
        # print(self.input_shape)
        self.model = self.build_model()

        # Train the model
        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=8,
            validation_split=0.2,
            verbose=1
        )

    def evaluate(self, training_set, test_set):
        """
        Evaluate the model on the test set.
        Args:
        - training_set: Training dataset.
        - test_set: Test dataset.
        """
        x_test, y_test = self.to_x_y(test_set)
        x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension if needed

        # Evaluate the model
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=1)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    def save_tflite(self):
        """
        Save the model in TFLite format and generate a flatbuffer.cpp file.
        """
        # Ensure the directory exists
        os.makedirs(self.path, exist_ok=True)
        
        # Convert the model to TFLite
        tflite_model_path = f"{self.path}/model.tflite"
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print(f"Model saved as TFLite at {tflite_model_path}")

        # Generate and save flatbuffer.cpp
        flatbuffer_path = f"{self.path}/flatbuffer.cpp"
        with open(flatbuffer_path, "w") as f:
            f.write('#include "flatbuffer.h"\n\n')
            f.write(f'int FlatBuffer::labelsLength = {len(self.labels)};\n\n')
            f.write('char *FlatBuffer::labels[] = {')
            f.write(', '.join([f'"{l}"' for l in self.labels]))
            f.write('};\n\n')
            f.write(f'int FlatBuffer::bytesLength = {len(tflite_model)};\n\n')
            f.write('unsigned char FlatBuffer::bytes[] = {\n   ')
            for i, val in enumerate(tflite_model):
                f.write(f'0x{val:02x}')
                if (i + 1) < len(tflite_model):
                    f.write(',')
                    if (i + 1) % 12 == 0:
                        f.write('\n   ')
                else:
                    f.write('\n')
            f.write('};\n')
        print(f"FlatBuffer saved as {flatbuffer_path}")


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
                self.train(list(zip(train_data, train_labels)), epochs=epochs)

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