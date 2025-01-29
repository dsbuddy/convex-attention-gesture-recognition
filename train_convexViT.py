import argparse
from data import Data
from model_convexViT import ConvexViT

# Set up argument parsing
parser = argparse.ArgumentParser(description="Train and evaluate a gesture recognition model.")  # Add description of the script
parser.add_argument(
    'gesture',  # Positional argument for the gesture type
    type=str,
    choices=['tap', 'swipe'],  # Restrict choices to valid gestures
    help="The gesture type to train the model on. Options are 'tap' or 'swipe'."  # Help message for the argument
)
args = parser.parse_args()  # Parse the command-line arguments

# Map gesture to the corresponding data filename
if args.gesture == 'tap':
    DATA_FILENAME = 'serial_MAC_NSEW_tap_ratio.json'  # File for tap gestures
elif args.gesture == 'swipe':
    DATA_FILENAME = 'serial_MAC_upDownLeftRight_swipe_ratio.json'  # File for swipe gestures

# Optional batch strategy (not used further but can be expanded in the future)
batch = 'pad'

# Load data from the specified JSON file
print('Loading trials...')
data = Data(DATA_FILENAME).load()  # Load data using the Data class
labels = data.labels()  # Extract labels from the data
training_set, test_set = data.split(0.3)  # Split the data into training and test sets (30% for testing)

# Train and evaluate the model
print('Training model...')
modelName = f'ConvexViT_{args.gesture}_{DATA_FILENAME.split(".")[0]}'  # Generate a name for the model based on the gesture and filename
model = ConvexViT(labels, modelName)  # Initialize the model with labels and the model name
model.train(training_set)  # Train the model on the training set
model.evaluate_on_training_and_test(training_set, test_set)  # Evaluate the model on both the training and test sets
model.evaluate_with_crossfold(data)  # Perform cross-validation on the entire dataset
