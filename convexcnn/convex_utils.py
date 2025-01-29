import numpy as np
from sklearn.metrics import f1_score

def evaluate_classifier(X_train, X_test, Y_train, Y_test, A, d2):
	n_train = X_train.shape[0]
	n_test = X_test.shape[0]

	# Calculate probabilities (assumes softmax-like output)
	eXAY = np.exp(np.sum((np.dot(X_train, A.T)) * Y_train[:,0:d2], axis=1)) 
	eXA_sum = np.sum(np.exp(np.dot(X_train, A.T)), axis=1) + 1
	loss = - np.average(np.log(eXAY/eXA_sum))

	# Make predictions 
	predictions_train = np.dot(X_train, A.T)
	predictions_test = np.dot(X_test, A.T)

	# Calculate error rates
	error_train = np.average(np.argmax(predictions_train, axis=1) != np.argmax(Y_train, axis=1))
	error_test = np.average(np.argmax(predictions_test, axis=1) != np.argmax(Y_test, axis=1))

	# Calculate accuracy
	acc_train = np.sum(np.argmax(predictions_train, axis=1) == np.argmax(Y_train, axis=1)) / n_train
	acc_test = np.sum(np.argmax(predictions_test, axis=1) == np.argmax(Y_test, axis=1)) / n_test

	# Calculate F1-score
	y_true_train = np.argmax(Y_train, axis=1)
	y_pred_train = np.argmax(predictions_train, axis=1)
	f1_train = f1_score(y_true_train, y_pred_train, average='macro')
	y_true_test = np.argmax(Y_test, axis=1)
	y_pred_test = np.argmax(predictions_test, axis=1)
	f1_test = f1_score(y_true_test, y_pred_test, average='macro')
	return loss, error_train, error_test, acc_train, acc_test, f1_train, f1_test

def project_to_nuclear_norm(A, R, P, nystrom_dim, d2):
	# Projects a matrix to a lower-rank approximation constrained by the nuclear norm
	# Reshape for SVD calculation
	A = A.reshape(d2 * P, nystrom_dim)
	# Perform Singular Value Decomposition (SVD)
	U, s, V = np.linalg.svd(A, full_matrices=False)
	# Project singular values onto L1-ball to constrain nuclear norm
	s = euclidean_proj_l1ball(s, s=R)
	# Reconstruct the projected matrix
	Ahat = np.reshape(np.dot(U, np.dot(np.diag(s), V)), (d2, P * nystrom_dim))
	return Ahat, U, s, V

def euclidean_proj_simplex(v, s=1):
	# Projects a vector onto the probability simplex
	assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
	n, = v.shape  # Ensure v is 1-D
	# Optimization: Check if already on simplex
	if v.sum() == s and np.alltrue(v >= 0):
		return v  
	# Sort 'v' descending, calculate cumulative sums
	u = np.sort(v)[::-1]
	cssv = np.cumsum(u)
	# Find index where condition is violated, compute theta
	rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
	theta = (cssv[rho] - s) / (rho + 1.0)
	# Project by subtracting theta and clipping at 0
	w = (v - theta).clip(min=0)
	return w

def euclidean_proj_l1ball(v, s=1):
	#Projects a vector onto the L1-ball
	assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
	n, = v.shape  # Ensure v is 1-D
	# Optimization: Check if already within the L1-ball
	if np.abs(v).sum() <= s:
		return v  
	# Project absolute values onto the simplex 
	u = np.abs(v)
	w = euclidean_proj_simplex(u, s)
	# Reconstruct the projection with original signs
	w *= np.sign(v)  
	return w

def calculate_attention(X, A):
	batch_size, input_dim = X.shape  # Get batch size and input dimension
	d2, seq_len_times_nystrom_dim = A.shape

	# Infer seq_len and nystrom_dim from A's shape
	seq_len = seq_len_times_nystrom_dim // input_dim
	nystrom_dim = input_dim // seq_len

	# Reshape A and X for compatibility
	A_reshaped = A.reshape(d2, seq_len, nystrom_dim)  # (d2, sequence_length, nystrom_dim)
	X_reshaped = X.reshape(batch_size, seq_len, nystrom_dim)

	# Calculate attention scores (unscaled)
	attention_scores = np.einsum("bsi,ksi->bsk", X_reshaped, A_reshaped) / np.sqrt(nystrom_dim)

	# Convex Combination (Softmax)
	attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=2, keepdims=True)

	# Ensure Convexity with additional constraints
	attention_weights = np.maximum(attention_weights, 0)  # Non-negativity constraint
	row_sums = np.sum(attention_weights, axis=2, keepdims=True)  # Row sum calculation for normalization
	attention_weights /= row_sums  # Normalize rows to sum to 1

	# return attention_weights
	return attention_weights.reshape(attention_weights.shape[0], attention_weights.shape[-1]).T

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