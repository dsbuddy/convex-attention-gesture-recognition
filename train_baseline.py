import argparse
from data import Data
from model_baseline import PadModel, PadAttentionModel

# Set up argument parsing
parser = argparse.ArgumentParser(description="Train and evaluate a gesture recognition model using MobileNet.")
parser.add_argument(
    'gesture',  # Positional argument for gesture type
    type=str,
    choices=['tap', 'swipe'],  # Restrict input to valid gestures
    help="The gesture type to train the model on. Options are 'tap' or 'swipe'."  # Help description
)
parser.add_argument(
    'model',  # Positional argument for model type
    type=str,
    choices=['cnn', 'attention'],  # Restrict input to valid model types
    help="The model type to use. Options are 'cnn' or 'attention'."  # Help description
)

args = parser.parse_args()  # Parse the command-line arguments

# Map gesture type to the corresponding data filename
if args.gesture == 'tap':
    DATA_FILENAME = 'serial_MAC_NSEW_tap_ratio.json'  # Filename for tap gesture
elif args.gesture == 'swipe':
    DATA_FILENAME = 'serial_MAC_upDownLeftRight_swipe_ratio.json'  # Filename for swipe gesture

# Optional batch strategy
batch = 'pad'

# Load the data
print('Loading trials...')
data = Data(DATA_FILENAME).load()  # Load data from the specified file
labels = data.labels()  # Extract labels from the dataset
training_set, test_set = data.split(0.3)  # Split data into training (70%) and test (30%) sets

# Train and evaluate the model
print('Training model...')
if args.model == 'cnn':
    modelName = f'baseline_cnn_{args.gesture}_{DATA_FILENAME.split(".")[0]}'  # Generate a model name based on gesture and filename
    model = PadModel(labels, modelName)  # Initialize the Baseline Attention model with labels and model name    

elif args.model == 'attention':
    modelName = f'baseline_attn_{args.gesture}_{DATA_FILENAME.split(".")[0]}'  # Generate a model name based on gesture and filename
    model = PadAttentionModel(labels, modelName)  # Initialize the Baseline Attention model with labels and model name    
    
model.train(training_set)  # Train the model on the training set
model.evaluate(training_set, test_set)  # Evaluate the model on the training and test sets
model.save_tflite()  # Convert and save the model to TensorFlow Lite format
model.evaluate_crossfold(data)  # Perform cross-validation on the dataset
