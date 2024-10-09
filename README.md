# Knitted Capacitive Touch Sensor Gesture Recognition

This repository contains the source code for research paper learning gesture recognition on a knitted capacitive touch sensor. The code provides functionality for data management, visualization, model implementation, and training for gesture recognition on knitted capacitive touch sensors.

## Traditional CNN Model

The following section of the readme details source code for a traditional CNN model. The code provides functionality for data processing, model implementation, and training for classification tasks.

### Data Management and Visualization

The codebase includes a set of classes for managing and visualizing trial data:

#### `Trial` Class

The `Trial` class represents a single trial with a label and associated data. It provides methods for creating, converting, and plotting trial instances.

#### `Data` Class

The `Data` class is responsible for managing a collection of `Trial` instances. It allows loading and saving trial data from/to JSON files, adding new trials, splitting the data into training and test sets, and visualizing the trial data using matplotlib.

### Model Implementation

The code defines a base class `_Model` and a subclass `PadModel` for training and evaluating machine learning models using the Keras library.

#### `_Model` Class

The `_Model` class serves as a base class for model implementations. It provides common functionality for loading, saving, evaluating models, converting trial data to model inputs and labels, training models, and making predictions.

#### `PadModel` Class

The `PadModel` class inherits from `_Model` and defines the architecture of a convolutional neural network (CNN) model for processing time-series data. The CNN architecture consists of convolutional, pooling, and dense layers, designed for gesture recognition on knitted capacitive touch sensors.

### Utilities

The repository includes a module with several utility functions for working with text data and numerical operations, such as loading and saving text files, one-hot encoding, and applying moving average filters.

### Training the Model

The `train.py` script is responsible for training and evaluating the machine learning model. It imports the necessary classes, loads the data, splits it into training and test sets, creates a model instance, and trains and evaluates the model.

To run the training process, you need to uncomment the lines for `model.train(training_set)` and `model.evaluate(training_set, test_set)` in the `train.py` script.

### Usage

1. Clone the repository
2. Install the required dependencies (e.g., Keras, NumPy, Matplotlib).
3. Prepare your data in the appropriate format (JSON files containing trial data).
4. Update the `DATA_FILENAME` variable in `train.py` to point to your data file.
5. Run the `train.py` script to train and evaluate the model.
6. Explore the provided classes and utilities for data management, visualization, and model implementation as needed.



## Convex CNN Model

The following section of the readme details source code for a convex CNN model. The code provides functionality for data processing, model implementation, and training for classification tasks.

### Data Processing

The `process_data` function in `train_convex.py` is responsible for loading and preprocessing data from a JSON file. It performs the following steps:

1.  Loads trial data from a JSON file.
2.  Extracts input features and corresponding labels.
3.  Shuffles the data randomly.
4.  Converts the data to NumPy arrays.
5.  Normalizes the input features by subtracting the mean and dividing by the standard deviation.

### Model Implementation

The `ConvexNeuralModel` class in `model_convex.py` defines the architecture and training procedure for the convex CNN model. It includes methods for:

-   Initializing the model with hyperparameters and data.
-   Constructing the Q matrix used in the convex optimization problem.
-   Training the model using a gradient descent algorithm.
-   Evaluating the model on training and test data.

The model also utilizes the `RandomFourierTransformer` class for feature transformation. This class applies a random Fourier feature mapping to the input data, which is a technique used to approximate kernel functions.


#### ConvexNeuralModel Class

The `ConvexNeuralModel` class implements a convexified convolutional neural network (CNN) for classification tasks. It leverages random Fourier features and a nuclear-norm constraint to achieve a convex optimization problem, which guarantees a globally optimal solution.

##### Key Operations and Functions

1.  **Initialization (`__init__`)**
    
    -   Stores the input data (`X`), labels (`Y`), and number of training samples (`n_train`).
    -   Calculates data properties such as the number of classes (`d2`), total number of samples (`n`), number of patches (`P`), and feature dimension of each patch (`d1`).
    -   Stores hyperparameters such as the Nyström dimension (`nystrom_dim`), gamma for the RBF kernel (`gamma`), nuclear norm radius (`R`), variance for dimensionality reduction (`variance`), learning rate (`learning_rate`), number of iterations (`n_iter`), mini-batch size (`mini_batch_size`), and number of mini-batches (`nr_of_mini_batches`).
    -   Sets up model saving for each epoch by creating a directory and defining the log file name.
    -   Constructs patches from the input data (`Z`).
2.  **Constructing Q (`construct_Q`)**
    
    -   Applies the `RandomFourierTransformer` to the input data to generate random Fourier features.
    -   Splits the data into training and test sets.
    -   Fits the transformer on the training data and transforms both training and test data.
    -   Reshapes the transformed data back into the original shape with the added Nyström dimension.
    -   Optionally performs feature normalization by subtracting the mean and dividing by the norm.
3.  **Training (`train`)**
    
    -   Converts the labels to binary format using `label_binarize`.
    -   Calls the `projected_gradient_descent` function to train the model.
    -   Returns the learned matrix A, filters, transformer, and data for the best model.
4.  **Projected Gradient Descent (`projected_gradient_descent`)**
    
    -   Reshapes the training and test data to flatten the input.
    -   Initializes the matrix A randomly.
    -   Performs projected stochastic gradient descent (PSGD) for a specified number of iterations.
    -   In each iteration, samples a mini-batch of data and calculates the gradient of the objective function.
    -   Updates the matrix A using the gradient and learning rate.
    -   Projects the updated A onto the nuclear norm ball using the `project_to_nuclear_norm` function.
    -   Evaluates the model on training and test data using the `evaluate_classifier` function.
    -   Keeps track of the best model based on test accuracy.
    -   Returns the learned matrix A, filters, and data for the best model.

#### Attention Function (`calculate_attention`)

The `calculate_attention` function calculates attention weights for the input data given the learned matrix A.

-   It reshapes the matrix A and input data X to be compatible with the attention calculation.
-   Calculates the attention scores using a scaled dot-product between A and X.
-   Applies a softmax function to the attention scores to obtain attention weights.
-   Ensures convexity by applying non-negativity and normalization constraints to the attention weights.
-   Returns the attention weights.

### Training the Model

The `train_convex.py` script is responsible for training and evaluating the convex CNN model. It includes functionality for:

-   Parsing command-line arguments for hyperparameters and data file.
-   Processing the data using the `process_data` function.
-   Initializing and training the `ConvexNeuralModel`.
-   Evaluating the trained model on training and test data.
-   Performing Bayesian optimization of hyperparameters using the `gp_minimize` function from the `skopt` library.


### Usage

1.  Clone the repository.
    
2.  Install the required dependencies (e.g., NumPy, Scikit-learn, Scikit-optimize).
    
3.  Prepare your data in the appropriate JSON format, ensuring it contains a list of trials, each with a "label" and corresponding "data".
    
4.  Run the script `train_convex.py` with the following command-line arguments:
    
    -   `--mode`: Specify the mode of operation. Use `learn` for Bayesian hyperparameter optimization, or any other value for manual hyperparameter setting.
    -   `--filename`: Path to your JSON data file.
    -   `--p`: Number of patches in the input data.
    -   `--d1`: Feature dimension of each patch.
    -   `--d2`: Number of classes (output dimension).
    -   `--r`: Nuclear norm radius for projection.
    -   `--variance`: Percentage of variance explained by the top singular values.
    -   `--nystrom_dim`: Nyström dimension for approximation.
    -   `--gamma`: Hyperparameter for the RBF kernel.
    -   `--lr`: Learning rate for gradient descent.
    -   `--train_test`: Fraction of data for training (default is 0.7).
    -   `--n_iter`: Number of iterations for gradient descent.
    -   `--mini_batch_size`: Size of mini-batches.
    -   `--nr_of_mini_batches`: Number of mini-batches per iteration.
    -   `--n_calls`: Number of calls to the objective function for Bayesian optimization (used in `learn` mode).
    -   `--use_attention`: Include this flag to enable attention mechanism.
    
    For example:
    

    
    ```bash
    python train_convex.py --mode learn --filename ./data/data_sample.json --p 10 --d1 6 --d2 4 --n_calls 100
    
    ```
    
    
    This command will run the script in Bayesian optimization mode (`--mode learn`) with the specified data file (`--filename ./data/data_sample.json`) and hyperparameters. The optimization process will evaluate the objective function 100 times (`--n_calls 100`).
    
5.  Explore the `model_convex.py` script for details on the model implementation and functions.

### Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License

This project is licensed under the [MIT License](LICENSE).

### Acknowledgments

We would like to thank the authors of the research paper and the contributors to the open-source libraries used in this project.
