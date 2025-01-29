import argparse
from data import Data
from model_squeezenet import SqueezeNetModel

# Set up argument parsing
parser = argparse.ArgumentParser(description="Train and evaluate a gesture recognition model using SqueezeNet.")
parser.add_argument(
    'gesture',  # Positional argument for gesture type
    type=str,
    choices=['tap', 'swipe'],  # Restrict input to 'tap' or 'swipe'
    help="The gesture type to train the model on. Options are 'tap' or 'swipe'."  # Description for the argument
)
args = parser.parse_args()  # Parse the command-line arguments

# Map gesture type to the appropriate data filename
if args.gesture == 'tap':
    DATA_FILENAME = 'serial_MAC_NSEW_tap_ratio.json'  # File for tap gestures
elif args.gesture == 'swipe':
    DATA_FILENAME = 'serial_MAC_upDownLeftRight_swipe_ratio.json'  # File for swipe gestures

# Load the data
print('Loading trials...')
data = Data(DATA_FILENAME).load()  # Load data from the specified file
labels = data.labels()  # Extract labels from the dataset
training_set, test_set = data.split(0.3)  # Split data into training (70%) and test (30%) sets

# Train and evaluate the model
print('Training model...')
modelName = f'squeezenet_{args.gesture}_{DATA_FILENAME.split(".")[0]}'  # Generate a name for the model based on gesture and filename
model = SqueezeNetModel(labels, modelName)  # Initialize the SqueezeNet model with labels and model name
model.train(training_set)  # Train the model using the training set
model.evaluate(training_set, test_set)  # Evaluate the model on the training and test sets
model.save_tflite()  # Convert and save the model to TensorFlow Lite format
model.evaluate_crossfold(data)  # Perform cross-validation evaluation
