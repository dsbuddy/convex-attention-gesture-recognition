import os
import csv
import json
from datetime import datetime
import math
import numexpr as ne
import numpy as np
import pickle
from sklearn.preprocessing import label_binarize

from convex_utils import calculate_attention, evaluate_classifier, project_to_nuclear_norm
from file_utils import dump_model_to_header

# Class Definition
class RandomFourierTransformer:
	transform_matrix = 0
	transform_bias = 0
	n_components = 0
	gamma = 0

	def __init__(self, gamma, n_components):
		self.n_components = n_components
		self.gamma = gamma

	def fit(self, X):
		d = X.shape[1]
		self.transform_matrix = np.random.normal(loc=0, scale=math.sqrt(2*self.gamma), size=(d, self.n_components)).astype(np.float32)
		self.transform_bias = (np.random.rand(1, self.n_components) * 2 * math.pi).astype(np.float32)

	def transform(self, Y):
		ny = Y.shape[0]
		angle = np.dot(Y, self.transform_matrix)
		bias = self.transform_bias
		factor = np.float32(math.sqrt(2.0 / self.n_components))
		# return ne.evaluate("factor*cos(angle+bias)")
		return factor * np.cos(angle+bias)

# Convexified Convolutional Neural Network Class
class ConvexNeuralModel:
	def __init__(self, X, Y, n_train, nystrom_dim, gamma, R, variance, learning_rate=0.1, n_iter=750, mini_batch_size=25, nr_of_mini_batches=50, hyperparameter_path_log='hyperparameter_log.csv', path=None, label_enum=None, data_filename=None, use_attention=False):
		# Storing data
		self.X_raw = X
		self.label = Y
		
		# Storing data properties
		self.d2 = np.unique(self.label).shape[0] # Output dimension (classes).
		self.n = self.X_raw.shape[0] # Number of patches frames.
		self.P = self.X_raw.shape[1] # Number of patch vectors.
		self.d1 = self.X_raw.shape[2] # Feature dimension of input patch.
		self.labels = label_enum # Enumerated labels (i.e. N, S, E, W).
		# Storing hyperparameters
		self.n_train = n_train
		self.n_test = self.n - self.n_train
		self.nystrom_dim = nystrom_dim # in Thesis: m
		self.gamma = gamma 
		self.R = R
		self.variance = variance
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.mini_batch_size = mini_batch_size
		self.nr_of_mini_batches = nr_of_mini_batches
		self.use_attention = use_attention

		# Set up model saving for each epoch
		self.path = path if path is not None else 'model_' + datetime.now().strftime('%Y%m%d_%H%M%S')
		os.makedirs(self.path, exist_ok=True)
		self.hyperparameter_log_filename = os.path.join(self.path, hyperparameter_path_log)
		self.data_filename = data_filename
		
		# Construct patches
		self.Z = self.X_raw.reshape(self.n, self.P, self.d1)
		print("Data contains " + str(self.n) + " samples, with " + str(self.P) + " patches of dimension " + str(self.d1) + ".") 
		print("Output contains " + str(self.n) + " samples, with " + str(self.d2) + " classes.") 

	def construct_Q(self, feature_normalization=True):
		print("Using Random Fourier Transformation")
		print("Creating Q...")

		Z_train = self.Z[0:self.n_train].reshape((self.n_train * self.P, self.d1))
		Z_test = self.Z[self.n_train:self.n].reshape((self.n_test * self.P, self.d1))
		self.transformer = RandomFourierTransformer(gamma=self.gamma, n_components=self.nystrom_dim)
		self.transformer.fit(X=Z_train)
		Q_train = self.transformer.transform(Z_train)
		Q_test = self.transformer.transform(Z_test)
		self.Q_train = Q_train.reshape((self.n_train, self.P, self.nystrom_dim))
		self.Q_test = Q_test.reshape((self.n_test, self.P, self.nystrom_dim))

		if feature_normalization == True:
			self.Q_train = self.Q_train.reshape((self.n_train * self.P, self.nystrom_dim))
			self.Q_train -= np.mean(self.Q_train, axis=0)
			self.Q_train /= np.linalg.norm(self.Q_train) / math.sqrt(self.n_train * self.P)
			self.Q_train = self.Q_train.reshape((self.n_train, self.P, self.nystrom_dim))
			self.Q_test = self.Q_test.reshape((self.n_test * self.P, self.nystrom_dim))
			self.Q_test -= np.mean(self.Q_test, axis=0)
			self.Q_test /= np.linalg.norm(self.Q_test) / math.sqrt(self.n_train * self.P)
			self.Q_test = self.Q_test.reshape((self.n_test, self.P, self.nystrom_dim))
	  
	def train(self, mean, std): 
		print("Training ConvexNeuralModel using projected stochastic gradient descent...")
		binary_label = label_binarize(self.label, classes=range(0, self.d2))
		self.mean = mean
		self.std = std
		self.Y_train=binary_label[0:self.n_train] 
		self.Y_test=binary_label[self.n_train:] 
		self.A, self.filter, best_model_data = self.projected_gradient_descent(X_train=self.Q_train, Y_train=self.Y_train, X_test=self.Q_test, Y_test=self.Y_test, n_iter=self.n_iter, R=self.R, variance=self.variance, learning_rate=self.learning_rate, d2=self.d2, mini_batch_size=self.mini_batch_size, nr_of_mini_batches=self.nr_of_mini_batches, transformer=self.transformer)
		return self.A, self.filter, self.transformer, best_model_data
	
	def projected_gradient_descent(self, X_train, Y_train, X_test, Y_test, R, variance, n_iter, learning_rate, d2, mini_batch_size, nr_of_mini_batches, transformer):  # Add path argument
		# Data Setup
		n_train, P, nystrom_dim = X_train.shape 
		n_test = X_test.shape[0]
		X_train = X_train.reshape(n_train, P*nystrom_dim)  # Flatten input data 
		X_test = X_test.reshape(n_test, P*nystrom_dim)

		# Initialization 
		best_train_acc = 0.0
		best_test_acc = 0.0
		best_train_f1 = 0.0
		best_test_f1 = 0.0
		best_model_data = None
		A = np.random.randn(d2, P*nystrom_dim)  # Initialize matrix A
		
		losses = []
		
		# Projected Stochastic Gradient Descent
		for t in range(n_iter):
			for _ in range(0, nr_of_mini_batches):
				# Sample a mini-batch
				index = np.random.randint(0, n_train, mini_batch_size) 
				X_sample = X_train[index]
				Y_sample = Y_train[index, :d2]

				# Stochastic Gradient Descent Update
				XA = np.dot(X_sample, A.T)
				eXA = ne.evaluate("exp(XA)")  # Assuming numerical expression evaluation
				eXA_sum = np.sum(eXA, axis=1).reshape((mini_batch_size, 1)) + 1
				diff = ne.evaluate("eXA/eXA_sum - Y_sample")  
				
				if self.use_attention:
					# Attention-Weighted Update
					attention_weights = calculate_attention(X_sample, A)  # Implement attention calculation
					grad_A = np.dot(diff.T * attention_weights, X_sample) / mini_batch_size  # Attention-weighted gradient
				else:
					grad_A = np.dot(diff.T, X_sample) / mini_batch_size
					
				A -= learning_rate * grad_A  

			# Projection to Nuclear Norm Constraint
			A, U, s, V = project_to_nuclear_norm(A, R, P, nystrom_dim, d2)
			# Dimensionality Threshold: The code calculates the fraction of cumulative variance explained by the top 25 singular values (dim = np.sum(s[0:25]) / np.sum(s))
			# This fraction provides a measure of how much information is captured by the first 25 components.
			# Variance Explained:  If dim is close to 1 (e.g., 0.95), it would mean that the top 25 components encompass 95% of the variance in the original data.
			# Consequently, you might choose to project your data onto these 25 components for dimensionality reduction, retaining most of the important information.
			dim = min(np.sum((s > 0).astype(int)), variance)

			# Model Evaluation
			loss, error_train, error_test, acc_train, acc_test, f1_train, f1_test = evaluate_classifier(X_train, X_test, Y_train, Y_test, A, d2)
			losses.append({'epoch': t, 'loss': loss, 'error_train': error_train, 'error_test': error_test, 'acc_train': acc_train, 'acc_test': acc_test, 'f1_train': f1_train, 'f1_test': f1_test})

			if acc_test > best_test_acc:
				best_train_acc = acc_train
				best_test_acc = acc_test
				best_train_f1 = f1_train
				best_test_f1 = f1_test
				best_epoch = t+1
				best_model_data = {
					"A": A,
					"filter": V[0:dim],
					"transformer": transformer,
				}
				
				# Save best model
				self.save_model(best_model_data["A"], best_model_data["filter"], best_model_data["transformer"], f"{best_epoch:04d}", f"{best_train_acc:.4f}", f"{best_test_acc:.4f}", f"{best_train_f1:.4f}", f"{best_test_f1:.4f}", filename="convex_model_model_BEST_MODEL.pkl")
				header_filename = os.path.join(self.path, 'model_data_EPOCH_{}_ACC_TRAIN_{}_ACC_TEST_{}_F1_TRAIN_{}_F1_TEST_{}.h'.format(f"{best_epoch:04d}", f"{best_train_acc:.4f}", f"{best_test_acc:.4f}", f"{best_train_f1:.4f}", f"{best_test_f1:.4f}"))
				dump_model_to_header(best_model_data, output_header_name=header_filename, mean=self.mean, std=self.std, labels=self.labels)
				print(f"Best Epoch: {best_epoch:04d} Train Acc: {best_train_acc:.4f}, Test Acc: {best_train_acc:.4f} Train F1: {best_train_f1:.4f}, Test F1: {best_test_f1:.4f}")


		complete_filename = os.path.join(self.path, 'losses.json')
		with open(complete_filename, "w") as loss_dump:
			json.dump(losses, loss_dump)
		
		return A, V[0:dim], best_model_data

	# Model Saving and Loading Functions
	def save_model(self, A, filter, transformer, epoch, acc_train, acc_test, f1_train, f1_test, filename="convex_model_model.pkl"):
		transform_matrix = transformer.transform_matrix
		transform_bias = transformer.transform_bias
		model_dict = {
			"A": A,
			"filter": filter,
			"transformer_matrix": transform_matrix,
			"transformer_bias": transform_bias,
			"transformer_n_components": transformer.n_components,
			"transformer_gamma": transformer.gamma,
		}

		self.log_best_model(acc_train, acc_test, f1_train, f1_test, epoch)
		# epoch_filename = filename.replace(".pkl", f"_epoch{epoch}_{acc_train:.4f}_{acc_test:.4f}.pkl")
		epoch_filename = "{}_epoch_{}_acc_train_{}_acc_test_{}.pkl".format(filename[:-4], epoch, acc_train, acc_test)
		complete_filename = os.path.join(self.path, epoch_filename)
		with open(complete_filename, 'wb') as f:
			pickle.dump(model_dict, f)

	def log_best_model(self, train_acc, test_acc, train_f1, test_f1, epoch):
		fieldnames = ["data_filename", "epoch", "train_acc", "test_acc", "train_f1", "test_f1", "r", "variance", "nystrom_dim", "gamma", "learning_rate", "n_iter", "mini_batch_size", "nr_of_mini_batches"]
		with open(self.hyperparameter_log_filename, 'a+', newline='') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writerow({
				'data_filename': self.data_filename,
				'epoch': epoch,
				'train_acc': train_acc,
				'test_acc': test_acc,
				'train_f1': train_f1,
				'test_f1': test_f1,
				'r': self.R,
				'variance': self.variance,
				'nystrom_dim': self.transformer.n_components, 
				'gamma': self.transformer.gamma,
				'learning_rate': self.learning_rate,
				'n_iter': self.n_iter,
				'mini_batch_size': self.mini_batch_size,
				'nr_of_mini_batches': self.nr_of_mini_batches
			})