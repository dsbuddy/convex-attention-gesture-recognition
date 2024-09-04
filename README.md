# Knitted Capacitive Touch Sensor Gesture Recognition

This repository contains the source code for the research paper "A Minimal Neural Network for Reproducible Gesture Recognition on Knitted Capacitive Touch Sensors". The code provides functionality for data management, visualization, model implementation, and training for gesture recognition on knitted capacitive touch sensors.

## Data Management and Visualization

The codebase includes a set of classes for managing and visualizing trial data:

### `Trial` Class

The `Trial` class represents a single trial with a label and associated data. It provides methods for creating, converting, and plotting trial instances.

### `Data` Class

The `Data` class is responsible for managing a collection of `Trial` instances. It allows loading and saving trial data from/to JSON files, adding new trials, splitting the data into training and test sets, and visualizing the trial data using matplotlib.

## Model Implementation

The code defines a base class `_Model` and a subclass `PadModel` for training and evaluating machine learning models using the Keras library.

### `_Model` Class

The `_Model` class serves as a base class for model implementations. It provides common functionality for loading, saving, evaluating models, converting trial data to model inputs and labels, training models, and making predictions.

### `PadModel` Class

The `PadModel` class inherits from `_Model` and defines the architecture of a convolutional neural network (CNN) model for processing time-series data. The CNN architecture consists of convolutional, pooling, and dense layers, designed for gesture recognition on knitted capacitive touch sensors.

## Utilities

The repository includes a module with several utility functions for working with text data and numerical operations, such as loading and saving text files, one-hot encoding, and applying moving average filters.

## Training the Model

The `train.py` script is responsible for training and evaluating the machine learning model. It imports the necessary classes, loads the data, splits it into training and test sets, creates a model instance, and trains and evaluates the model.

To run the training process, you need to uncomment the lines for `model.train(training_set)` and `model.evaluate(training_set, test_set)` in the `train.py` script.

## Usage

1. Clone the repository: `git clone https://github.com/dsbuddy/knitted-capacitive-touch-sensor-gesture-recognition.git`
2. Install the required dependencies (e.g., Keras, NumPy, Matplotlib).
3. Prepare your data in the appropriate format (JSON files containing trial data).
4. Update the `DATA_FILENAME` variable in `train.py` to point to your data file.
5. Run the `train.py` script to train and evaluate the model.
6. Explore the provided classes and utilities for data management, visualization, and model implementation as needed.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to thank the authors of the research paper and the contributors to the open-source libraries used in this project.
