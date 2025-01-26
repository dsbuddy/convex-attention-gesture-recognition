from data import Data
from model_convex import ConvexNeuralModel  # Import the model
import csv
import argparse
from sklearn.metrics import f1_score
from datetime import datetime
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import os
import json
import random
import numpy as np

# Data Processing
def process_data(filename, labels, shuffle=True):
	print('Loading trials...')
	with open(filename, 'r') as f:
		data = json.load(f)

	inputs_raw = [trial['data'] for trial in data]
	outputs_raw = [labels.index(trial['label']) for trial in data]

	# Shuffle data for better training
	combined = list(zip(inputs_raw, outputs_raw))
	if shuffle:
		random.shuffle(combined)
	inputs_raw, outputs_raw = zip(*combined) 

	# Convert to NumPy arrays for efficient processing
	inputs = np.array(inputs_raw)
	outputs = np.array(outputs_raw)
	
	mean = inputs.mean(axis=0)
	std = inputs.std(axis=0, ddof=1) + 1

	# Normalize inputs
	inputs = (inputs - mean) / (std)
	return inputs, outputs, mean, std

# Evaluation Functions
def transform_inference_sample(sample, transformer, nystrom_dim):  # Remove 'p' argument
	sample_transformed = transformer.transform(sample)
	p = sample.shape[0]  # Infer the number of patches from the sample
	sample_transformed = sample_transformed.reshape(1, p * nystrom_dim)
	return sample_transformed

def evaluate_model(x, y, A, transformer, p, nystrom_dim):
	correct = 0
	count = 0
	y_pred = []
	y_true = []
	for (x_sample, y_sample) in zip(x, y):
		x_sample_transformed = transform_inference_sample(x_sample, transformer, nystrom_dim)
		predicted = np.dot(x_sample_transformed, A.T)
		if np.argmax(predicted) == y_sample:
			correct += 1
		count += 1
		y_pred.append(np.argmax(predicted))
		y_true.append(y_sample)
	f1 = f1_score(y_true, y_pred, average='macro')
	return correct/count, f1

def arg_parse():
	parser = argparse.ArgumentParser(description="Graph Matching arguments.")
	#io_parser = parser.add_mutually_exclusive_group(required=False)
	parser.add_argument("--mode", type=str, default='learn', help="Mode to run the script in: 'learn' to automate hyperparameter search or anything else to set hyperparamters manually.")
	parser.add_argument("--filename", type=str, help="Input dataset json filename.")
	parser.add_argument("--p", dest="p", type=int, help="Number of patch vectors.")
	parser.add_argument("--d1", dest="d1", type=int, help="Feature dimension of input patch.")
	parser.add_argument("--d2", dest="d2", type=int, help="Output dimension [number of classes].")
	parser.add_argument("--r", dest="r", type=float, help="Nuclear Norm radius to project A on: ||A|| {∗} = R.")
	parser.add_argument("--variance", dest="variance", type=int, help="Percentage of variance explained by the top [variance] singular values; can be used to monitor how much information is being retained after the projection.")
	parser.add_argument("--nystrom_dim", dest="nystrom_dim", type=int, help="Nystroem dimension used to approximate Q during step 1 of Algorithm - In Thesis: ”m”.")
	parser.add_argument("--gamma", dest="gamma", type=float, help="Hyperparameter for the RBF kernel..")
	parser.add_argument('--lr', dest="lr", type=float, help="Step size at which a model adjusts its parameters during Stochastic Gradeint Descent.")
	parser.add_argument('--train_test', dest="train_test", type=float, help="Fraction of inputs to split to train and test.")
	parser.add_argument('--n_iter', dest="n_iter", type=int, help="Number of iterations for the Projected Stochastic Gradient Descent.")
	parser.add_argument('--mini_batch_size', dest="mini_batch_size", type=int, help="Number of iterations for the Projected Stochastic Gradient Descent.")
	parser.add_argument('--nr_of_mini_batch_size', dest="nr_of_mini_batches", type=int, help="Number of iterations for the Projected Stochastic Gradient Descent.")
	parser.add_argument('--n_calls', dest="n_calls", type=int, help="Number of calls to the objective function for Bayesian Optimization.")
	parser.add_argument('--use_attention', default=False, action="store_true", help="Disable attention by default")
	parser.set_defaults(
		filename='./data/data_sample.json',
		p=10,
		d1=6,
		d2=4,
		r=8.04558268,
		variance=57,
		nystrom_dim=3,
		gamma=3.71877749,
		lr=0.07114879,
		train_test_split=0.7,
		n_iter=50,
		mini_batch_size=32,
		nr_of_mini_batches=130,
		label_enum=['up', 'down', 'left', 'right'],
		n_calls=100,
		mode='train',
	)
	return parser.parse_args()

def learn_params_bayes(args, inputs, outputs, num_train):
	# Hyperparameter Search Space
	space = [
		Real(0.5, 10, name='R'),
		Integer(10, 75, name='variance'),
		Real(0.1, 10, name='gamma'),
		Real(0.001, 0.1, name='learning_rate'),
		Integer(10, 1000, name='n_iter'),
		Integer(10, 150, name='mini_batch_size'),
		Integer(10, 150, name='nr_of_mini_batches'),
		# Integer(10, 200, name='nystrom_dim'),
	]

	# Single Output Directory for Models
	output_path = 'model_' + datetime.now().strftime('%Y%m%d_%H%M%S')
	os.makedirs(output_path, exist_ok=True)  

	# Logging setup
	hyperparameter_log_filename = os.path.join(output_path, "hyperparameter_log.csv")
	with open(hyperparameter_log_filename, 'w', newline='') as csvfile:
		fieldnames = ["data_filename", "epoch", "train_acc", "test_acc", "train_f1", "test_f1", "r", "variance", "nystrom_dim", "gamma", "learning_rate", "n_iter", "mini_batch_size", "nr_of_mini_batches"]
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

	# Objective Function for Bayesian Optimization
	@use_named_args(space)
	def objective(**params):
		# print(params)  # Log hyperparameters
		if args.use_attention:
			model = ConvexNeuralModel(X=inputs, Y=outputs, n_train=num_train, path=output_path, data_filename=args.filename, nystrom_dim=args.nystrom_dim, use_attention=True, **params)
		else:
			model = ConvexNeuralModel(X=inputs, Y=outputs, n_train=num_train, path=output_path, data_filename=args.filename, nystrom_dim=args.nystrom_dim, use_attention=False, **params)
		model.construct_Q()
		A, filters, transformer, _ = model.train()
		train_acc, train_f1 = evaluate_model(inputs[:num_train], outputs[:num_train], A, model.transformer, args.p, args.nystrom_dim) # params['nystrom_dim']) # args.nystrom_dim) # params['nystrom_dim'])
		test_acc, test_f1 = evaluate_model(inputs[num_train:], outputs[num_train:], A, model.transformer, args.p, args.nystrom_dim) # params['nystrom_dim']) # args.nystrom_dim) #  params['nystrom_dim'])

		# Weighted Score (Prioritize Test Metrics)
		test_weight = 2 
		score = (train_acc + train_f1 + test_weight * test_acc + test_weight * test_f1) / (2 + 2 * test_weight)
		return -score  # Minimize the negative score to maximize the original score

	# Bayesian Optimization
	res_gp = gp_minimize(objective, space, n_calls=args.n_calls, random_state=0)  # # n_calls: the number of evaluations of the objective function to perform

	# Get and log the best hyperparameters and score
	best_params = {param.name: res_gp.x[i] for i, param in enumerate(space)}
	best_score = -res_gp.fun  # The objective function returns the negative score

	print("Best Hyperparameters:", best_params)
	print("Best Score:", best_score)

def main():
	args = arg_parse()  # Parse arguments

	# Load Data
	inputs, outputs, _, _ = process_data(args.filename, args.label_enum)  # Load using process_data
	num_train = int(len(inputs) * args.train_test_split)

	if args.mode == 'learn':
		learn_params_bayes(args, inputs, outputs, num_train)
	else:
		# Example of direct training:
		inputs, outputs, mean, std = process_data(args.filename, args.label_enum)
		num_train = int(len(inputs)*args.train_test_split)

		if args.use_attention:
			model = ConvexNeuralModel(X=inputs, Y=outputs, n_train=num_train, data_filename=args.filename, nystrom_dim=args.nystrom_dim, use_attention=True,
								 R=args.r, variance=args.variance, gamma=args.gamma, learning_rate=args.lr, n_iter=args.n_iter, mini_batch_size=args.mini_batch_size, nr_of_mini_batches=args.nr_of_mini_batches)
		else:
			model = ConvexNeuralModel(X=inputs, Y=outputs, n_train=num_train, data_filename=args.filename, nystrom_dim=args.nystrom_dim, use_attention=True,
								 R=args.r, variance=args.variance, gamma=args.gamma, learning_rate=args.lr, n_iter=args.n_iter, mini_batch_size=args.mini_batch_size, nr_of_mini_batches=args.nr_of_mini_batches)
		model.construct_Q()
		A, filters, transformer, best_model = model.train(mean, std)
		train_acc, train_f1 = evaluate_model(inputs[:num_train], outputs[:num_train], A, model.transformer, args.p, args.nystrom_dim)
		test_acc, test_f1 = evaluate_model(inputs[num_train:], outputs[num_train:], A, model.transformer, args.p, args.nystrom_dim)
		print(best_model.keys())
		print("Train Acc: {:.4f}, Test Acc: {:.4f} Train F1: {:.4f}, Test F1: {:.4f}".format(train_acc, test_acc, train_f1, test_f1))

if __name__ == '__main__':
	main()