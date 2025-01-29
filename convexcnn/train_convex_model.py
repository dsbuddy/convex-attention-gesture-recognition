import os
import csv
import random
from datetime import datetime
import argparse
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from convex_model import ConvexNeuralModel
from convex_utils import evaluate_model
from file_utils import process_data

random.seed(0)

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Graph Matching arguments.")
    parser.add_argument("--mode", type=str, default='manual', help="Mode: 'learn' or manual setup.")
    # parser.add_argument("--filename", type=str, default='data.json', help="Dataset filename.")
    parser.add_argument(
        'gesture',  # Positional argument for gesture type
        type=str,
        choices=['tap', 'swipe'],  # Restrict input to 'tap' or 'swipe'
        help="The gesture type to train the model on. Options are 'tap' or 'swipe'."  # Description for the argument
    )
    parser.add_argument("--r", type=float, default=8.0, help="Nuclear norm radius ||A|| = R.")
    parser.add_argument("--variance", type=int, default=5, help="Variance explained percentage.")
    parser.add_argument("--nystrom_dim", type=int, default=3, help="Nystroem dimension (m).")
    parser.add_argument("--gamma", type=float, default=3.0, help="RBF kernel hyperparameter.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for SGD.")
    parser.add_argument("--train_test_split", type=float, default=0.7, help="Train-test split ratio.")
    parser.add_argument("--n_iter", type=int, default=50, help="Number of SGD iterations.")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--nr_of_mini_batches", type=int, default=128, help="Mini-batch count.")
    parser.add_argument("--n_calls", type=int, default=100, help="Bayesian optimization calls.")
    parser.add_argument("--use_attention", action="store_true", default=False, 
                        help="Enable attention mechanism.")
    return parser.parse_args()

# Bayesian Optimization
def optimize_hyperparameters(args, inputs, outputs, mean, std, output_path):
    space = [
        Real(0.5, 10, name='R'),
        Integer(10, 75, name='variance'),
        Real(0.1, 10, name='gamma'),
        Real(0.001, 0.1, name='learning_rate'),
        Integer(10, 1000, name='n_iter'),
        Integer(10, 150, name='mini_batch_size'),
        Integer(10, 150, name='nr_of_mini_batches')
    ]

    log_file = os.path.join(output_path, "hyperparameter_log.csv")
    with open(log_file, 'w', newline='') as csvfile:
        fieldnames = ["data_filename", "epoch", "train_acc", "test_acc", 
                      "train_f1", "test_f1", "R", "variance", 
                      "nystrom_dim", "gamma", "learning_rate", 
                      "n_iter", "mini_batch_size", "nr_of_mini_batches"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    @use_named_args(space)
    def objective(**params):
        model = ConvexNeuralModel(
            X=inputs, Y=outputs, n_train=num_train, path=output_path, 
            data_filename=DATA_FILENAME, nystrom_dim=args.nystrom_dim, 
            use_attention=args.use_attention, label_enum=labels, **params
        )
        model.construct_Q()
        A, _, transformer, _ = model.train(mean, std)
        train_acc, train_f1 = evaluate_model(inputs[:num_train], outputs[:num_train], 
                                             A, transformer, model.P, args.nystrom_dim)
        test_acc, test_f1 = evaluate_model(inputs[num_train:], outputs[num_train:], 
                                           A, transformer, model.P, args.nystrom_dim)
        # Weighted scoring prioritizing test metrics
        score = (train_acc + train_f1 + 2 * (test_acc + test_f1)) / 6
        return -score  # Minimize the negative score

    res_gp = gp_minimize(objective, space, n_calls=args.n_calls, random_state=0)
    best_params = {param.name: res_gp.x[i] for i, param in enumerate(space)}
    best_score = -res_gp.fun

    print("Best Hyperparameters:", best_params)
    print("Best Score:", best_score)

# Main Execution
if __name__ == '__main__':
    args = parse_arguments()
    mode_subdir = "hyp_search" if args.mode == "learn" else "trial"
    opt_subdir = "attn" if args.use_attention else "no_attn"

    # Map gesture type to the appropriate data filename
    if args.gesture == 'tap':
        DATA_FILENAME = 'serial_MAC_NSEW_tap_ratio.json'  # File for tap gestures
    elif args.gesture == 'swipe':
        DATA_FILENAME = 'serial_MAC_upDownLeftRight_swipe_ratio.json'  # File for swipe gestures

    output_path = os.path.join(
        '..', 'repo', 'models', 
        f"convex_model_{mode_subdir}_{DATA_FILENAME.split('.')[0]}_{opt_subdir}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_path, exist_ok=True)

    inputs, outputs, mean, std, labels = process_data(DATA_FILENAME)
    num_train = int(len(inputs) * args.train_test_split)

    if args.mode == 'learn':
        optimize_hyperparameters(args, inputs, outputs, mean, std, output_path)
    else:
        model = ConvexNeuralModel(
            X=inputs, Y=outputs, n_train=num_train, data_filename=DATA_FILENAME, 
            nystrom_dim=args.nystrom_dim, use_attention=args.use_attention,
            R=args.r, variance=args.variance, gamma=args.gamma, 
            learning_rate=args.lr, n_iter=args.n_iter, 
            mini_batch_size=args.mini_batch_size, nr_of_mini_batches=args.nr_of_mini_batches, 
            path=output_path, label_enum=labels
        )
        model.construct_Q()
        A, _, transformer, _ = model.train(mean, std)
        train_acc, train_f1 = evaluate_model(inputs[:num_train], outputs[:num_train], 
                                             A, transformer, model.P, args.nystrom_dim)
        test_acc, test_f1 = evaluate_model(inputs[num_train:], outputs[num_train:], 
                                           A, transformer, model.P, args.nystrom_dim)
        print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
              f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")