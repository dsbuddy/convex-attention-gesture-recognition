import math
import numpy as np
import os
import random
import torch
import torch.fft
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

from sklearn.metrics import f1_score
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils.data import DataLoader, Dataset

# TransformerBlock serves as a base class for Transformer-like architectures.
# This aligns with the theoretical foundation laid in "Unraveling Attention via Convex Duality" [ConvexViT].
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

# ConvexMLPBlock implements a convex optimization-inspired MLP block.
# The Burer-Monteiro decomposition is highlighted as a means to ensure global optimality, as discussed in Section 2.1 of the paper.
class ConvexMLPBlock(TransformerBlock):
    def __init__(self, dim, neurons=50, num_classes=10, h=14, activation='linear', burer_monteiro=False, burer_dim=10):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.HW = h * h
        self.act = activation

        self.embed_dim = neurons if activation == 'relu' else 1

        if activation == 'relu':
            # Arrangement generator aligns with the hyperplane arrangement in Section 2.1.
            self.arrangement_generator = nn.Linear(dim, self.embed_dim)
            self.arrangement_generator.weight.requires_grad = False
            self.arrangement_generator.bias.requires_grad = False
        else:
            self.arrangement_generator = None  # Set to None when not used

        self.burer = burer_monteiro
        if burer_monteiro:
            # Decomposition of W and V for efficient optimization, as described in Burer-Monteiro [ConvexViT, Section 2.1].
            self.W = torch.nn.Parameter(torch.randn(self.embed_dim, dim, burer_dim) / burer_dim, requires_grad=True)
            self.V = torch.nn.Parameter(torch.randn(self.embed_dim, burer_dim, num_classes) / burer_dim, requires_grad=True)
        else:
            self.W = None  # Placeholder when not used
            self.V = None  # Placeholder when not used
            self.linear_map = nn.Linear(dim, self.embed_dim * num_classes, bias=False)

    def forward(self, x):
        B, HW, D = x.shape

        if self.act == 'relu' and self.arrangement_generator is not None:
            with torch.no_grad():
                # Sign patterns correspond to activation patterns of ReLU MLPs (Section 2.1).
                sign_patterns = (self.arrangement_generator(x) > 0).float()  # B x HW x P

            if self.burer:
                assert self.W is not None and self.V is not None, "W and V must be defined when using burer_monteiro."
                preds = 1 / HW * torch.einsum('bhp, bhd, pdk -> pbk', sign_patterns, x, self.W)
                preds = torch.einsum('pbk, pkc -> bc', preds, self.V)
            else:
                preds = 1 / HW * torch.einsum(
                    'bhp, bhpc -> bc',
                    sign_patterns,
                    self.linear_map(x).reshape((B, HW, self.embed_dim, self.num_classes)),
                )
        else:
            if self.burer:
                assert self.W is not None and self.V is not None, "W and V must be defined when using burer_monteiro."
                preds = 1 / HW * torch.einsum('bhd, pdk -> bpk', x, self.W)
                preds = torch.einsum('bpk, pkc -> bc', preds, self.V)
            else:
                preds = 1 / HW * torch.sum(
                    self.linear_map(x).reshape((B, HW, self.embed_dim, self.num_classes)), (1, 2)
                )

        return preds / self.embed_dim

# TransformerTransfer combines Transformer blocks with pre-trained embeddings, inspired by Section 5.
class TransformerTransfer(nn.Module):
    def __init__(self, block, img_size=224, patch_size=16, num_classes=10, embed_dim=768, neurons=50, 
            drop_rate=0., drop_path=0., norm_layer=nn.Identity, dropcls=0., activation='linear',
            pooling=False, burer_monteiro=False, pooling_dim=100):
        super().__init__()
        self.num_classes = num_classes
        h = img_size // patch_size

        if pooling:
            embed_dim = pooling_dim

        burer_dim = embed_dim // 2
        if block == ConvexMLPBlock:
            burer_dim *= h * h

        self.embed_dim = embed_dim
        self.pooler = nn.AdaptiveAvgPool1d(self.embed_dim)
        self.block = block(dim=embed_dim, neurons=neurons, num_classes=num_classes, h=h, 
                           activation=activation, burer_monteiro=burer_monteiro, burer_dim=burer_dim)

    def forward(self, x):
        x = self.pooler(x).detach()
        outputs = self.block(x)
        return outputs

# Custom dataset for handling trials, mapping to TorchDataset for PyTorch training pipelines.
class TorchDataset(Dataset):
    def __init__(self, trials, labels):
        self.data = [trial.data for trial in trials]  # Access the `data` attribute
        self.labels = [labels.index(trial.label) for trial in trials]  # Access the `label` attribute

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ConvexViT encapsulates the architecture and training logic.
# It follows the convex principles and transfer learning techniques outlined in Sections 3 and 5.
class ConvexViT:

    MODEL_DIR = 'repo'
    MODEL_CONFIG = 'model.json'
    MODEL_WEIGHTS = 'model.weights.h5'

    def __init__(self, labels, filename):
        """
        Initialize the ConvexViT.

        Args:
            labels: List of class labels.
            neurons: Number of neurons in ConvexMLPBlock.
        """
        self.labels = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name = 'model_' + filename
        self.path = os.path.join(self.MODEL_DIR, 'models', name)
        os.makedirs(self.path, exist_ok=True)

    def train(self, trials, epochs=50, batch_size=32, lr=1e-3):
        """
        Train the model using convex-inspired principles.

        Args:
            trials: Dataset containing input signals and labels.
            epochs: Number of epochs.
            batch_size: Batch size.
            lr: Learning rate.
        """
        print(len(trials[0].data))
        # Extract sequence length from the first trial
        seq_len = len(trials[0].data)
        self.model = TransformerTransfer(
            block=ConvexMLPBlock,
            img_size=seq_len,              # Sequence length
            patch_size=seq_len,             # No splitting (1 patch = 1 timestep)
            num_classes=len(self.labels),  # Number of output classes
            embed_dim=8,              # Feature dimension
            neurons=50,          # Neurons in ConvexMLPBlock
        ).to(self.device)
        # exit()
        dataset = TorchDataset(trials, self.labels)
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            all_labels = []
            predictions = []
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Collect labels and predictions for F1 score computation
                all_labels.extend(labels.cpu().numpy())
                predictions.extend(predicted.cpu().numpy())
                train_f1 = f1_score(all_labels, predictions, average='macro')

            train_acc = 100. * correct / total
            val_loss, val_acc, val_f1 = self.evaluate(val_loader, criterion)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.2f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f} ")

    def evaluate(self, data_loader, criterion):
        """
        Evaluate the model performance.

        Args:
            data_loader: DataLoader for evaluation.
            criterion: Loss function.

        Returns:
            Tuple of validation loss, accuracy, and F1 score.
        """
        self.model.eval()
        loss = 0
        correct = 0
        total = 0
        all_labels = []
        predictions = []

        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                outputs = self.model(data)
                loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                predictions.extend(predicted.cpu().numpy())

        acc = 100. * correct / total
        macro_f1 = f1_score(all_labels, predictions, average='macro')
        return loss / len(data_loader), acc, macro_f1

    def evaluate_on_training_and_test(self, training_set, test_set, batch_size=32):
        """
        Evaluate the model on both the training and test datasets.

        Args:
            training_set: Dataset for training evaluation.
            test_set: Dataset for testing evaluation.
            batch_size: Batch size for DataLoader.

        Returns:
            A dictionary containing training and testing losses and accuracies.
        """
        # Create DataLoaders for training and test sets
        train_loader = DataLoader(TorchDataset(training_set, self.labels), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TorchDataset(test_set, self.labels), batch_size=batch_size, shuffle=False)

        # Use the same loss criterion as in training
        criterion = nn.CrossEntropyLoss()

        # Evaluate on the training set
        train_loss, train_acc, train_f1 = self.evaluate(train_loader, criterion)
        print(f"Training Set | Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, F1: {train_f1:.2f}")

        # Evaluate on the test set
        test_loss, test_acc, test_f1 = self.evaluate(test_loader, criterion)
        print(f"Test Set | Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, F1: {test_f1:.2f}")

        return {
            "training_loss": train_loss,
            "training_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        }

    def evaluate_with_crossfold(self, dataset, num_folds=10, batch_size=32):
        """
        Perform 10-fold cross-validation evaluation on the dataset.

        Args:
            dataset: Combined dataset containing both training and test data.
            num_folds: Number of cross-validation folds (default is 10).
            batch_size: Batch size for DataLoader.

        Returns:
            A dictionary containing average training and testing losses and accuracies across folds.
        """
        # Ensure the dataset is iterable
        trials = dataset.trials
        dataset_size = len(trials)
        fold_size = dataset_size // num_folds
        indices = list(range(dataset_size))
        random.shuffle(indices)

        fold_results = {
            "training_loss": [],
            "training_accuracy": [],
            "training_f1": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_f1": []
        }

        criterion = nn.CrossEntropyLoss()

        for fold in range(num_folds):
            print(f"Fold {fold + 1}/{num_folds}")

            # Define training and test splits
            test_indices = indices[fold * fold_size: (fold + 1) * fold_size]
            train_indices = list(set(indices) - set(test_indices))

            train_trials = [trials[i] for i in train_indices]
            test_trials = [trials[i] for i in test_indices]

            train_loader = DataLoader(TorchDataset(train_trials, self.labels), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(TorchDataset(test_trials, self.labels), batch_size=batch_size, shuffle=False)

            # Train the model on the current fold's training set
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            for epoch in range(1):  # Use 1 epoch per fold for simplicity
                for data, labels in train_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Save the FlatBuffer after training
            self.save_flatbuffer(self.labels)
            
            # Evaluate on the training set
            train_loss, train_acc, train_f1 = self.evaluate(train_loader, criterion)
            fold_results["training_loss"].append(train_loss)
            fold_results["training_accuracy"].append(train_acc)
            fold_results["training_f1"].append(train_f1)

            # Evaluate on the test set
            test_loss, test_acc, test_f1 = self.evaluate(test_loader, criterion)
            fold_results["test_loss"].append(test_loss)
            fold_results["test_accuracy"].append(test_acc)
            fold_results["test_f1"].append(test_f1)

            print(f"Fold {fold + 1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.2f}, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.2f} ")

        # Compute averages across folds
        avg_results = {key: np.mean(values) for key, values in fold_results.items()}

        print("\nCross-Fold Results:")
        print(f"Average Training Loss: {avg_results['training_loss']:.4f}, "
            f"Average Training Accuracy: {avg_results['training_accuracy']:.2f}%")
        print(f"Average Test Loss: {avg_results['test_loss']:.4f}, "
            f"Average Test Accuracy: {avg_results['test_accuracy']:.2f}%")

        return avg_results


    def save_flatbuffer(self, labels):
        """
        Save the PyTorch model and labels as a FlatBuffer format.
        
        Args:
            labels: List of class labels.
        """
        # Convert the PyTorch model to TorchScript
        scripted_model = torch.jit.script(self.model)
        model_buffer = scripted_model.save_to_buffer()

        # Define output paths
        flatbuffer_path = os.path.join(self.path, 'flatbuffer.cpp')
        labels_length = len(labels)
        model_length = len(model_buffer)

        # Write FlatBuffer C++ file
        with open(flatbuffer_path, 'w') as f:
            f.write('#include "flatbuffer.h"\n\n')
            f.write(f'int FlatBuffer::labelsLength = {labels_length};\n\n')
            f.write('char *FlatBuffer::labels[] = {')
            f.write(', '.join([f'"{label}"' for label in labels]))
            f.write('};\n\n')
            f.write(f'int FlatBuffer::bytesLength = {model_length};\n\n')
            f.write('unsigned char FlatBuffer::bytes[] = {\n   ')
            for i, val in enumerate(model_buffer):
                f.write(f'0x{val:02x}')
                if (i + 1) < model_length:
                    f.write(',')
                    if (i + 1) % 12 == 0:
                        f.write('\n   ')
                else:
                    f.write('\n')
            f.write('};\n')

        print(f"FlatBuffer saved to {flatbuffer_path}")