# Knitted Capacitive Touch Sensor Gesture Recognition

## Overview
This repository contains the source code for the research paper on gesture recognition using a knitted capacitive touch sensor. It includes scripts and utilities for data management, model training, and evaluation, targeting multiple architectures including Baseline CNN, MobileNet, SqueezeNet, and Convex ViT.

## Contents

### 1. Baseline CNN Model

**Training Script:** `train_baseline.py`

Trains a baseline CNN model with optional attention-based enhancement.

**Usage:**
```bash
python train_baseline.py <gesture> <model>
```
- `<gesture>`: Specify the gesture type (`tap` or `swipe`).
- `<model>`: Choose the model type (`cnn` or `attention`).

**Example:**
```bash
python train_baseline.py tap cnn
```

**Script Features:**
- Maps the gesture type to the corresponding dataset file.
- Trains the specified model (`cnn` or `attention`) on the dataset.
- Evaluates the model on training and test sets.
- Saves the model in TensorFlow Lite format.
- Performs cross-validation.

---

### 2. Convex ViT Model

**Training Script:** `train_convexViT.py`

Trains a Convex Vision Transformer (ViT) for gesture recognition.

**Usage:**
```bash
python train_convexViT.py <gesture>
```
- `<gesture>`: Specify the gesture type (`tap` or `swipe`).

**Example:**
```bash
python train_convexViT.py swipe
```

**Script Features:**
- Maps the gesture type to the appropriate dataset file.
- Trains the Convex ViT model on the dataset.
- Evaluates the model on training and test sets.
- Performs cross-validation.

---

### 3. MobileNet Model

**Training Script:** `train_mobilenet.py`

Trains a MobileNet-based model for gesture recognition.

**Usage:**
```bash
python train_mobilenet.py <gesture>
```
- `<gesture>`: Specify the gesture type (`tap` or `swipe`).

**Example:**
```bash
python train_mobilenet.py tap
```

**Script Features:**
- Maps the gesture type to the appropriate dataset file.
- Trains the MobileNet model on the dataset.
- Evaluates the model on training and test sets.
- Saves the model in TensorFlow Lite format.
- Performs cross-validation.

---

### 4. SqueezeNet Model

**Training Script:** `train_squeezenet.py`

Trains a SqueezeNet-based model for gesture recognition.

**Usage:**
```bash
python train_squeezenet.py <gesture>
```
- `<gesture>`: Specify the gesture type (`tap` or `swipe`).

**Example:**
```bash
python train_squeezenet.py swipe
```

**Script Features:**
- Maps the gesture type to the appropriate dataset file.
- Trains the SqueezeNet model on the dataset.
- Evaluates the model on training and test sets.
- Saves the model in TensorFlow Lite format.
- Performs cross-validation.

---

### 5. Convex CNN Model

**Training Script:** `train_convex.py`

Trains a Convex CNN model by mapping the gesture type to the appropriate dataset file.

**Usage:**
```bash
python train_convex.py <gesture> [--mode <mode>] [--use_attention]
```
- `<gesture>`: Specify the gesture type (`tap` or `swipe`).
- `--mode`: Choose between `learn` for Bayesian hyperparameter optimization or `manual` for direct parameter setting (default: `manual`).
- `--use_attention`: Add this flag to enable attention mechanisms.

**Example:**
```bash
python train_convex.py tap --mode learn --use_attention
```

**Script Features:**
- Maps the gesture type (`tap` or `swipe`) to a dataset file:
  - `tap`: `serial_MAC_NSEW_tap_ratio.json`
  - `swipe`: `serial_MAC_upDownLeftRight_swipe_ratio.json`
- Trains the Convex CNN model with options for Bayesian optimization.
- Evaluates the model on training and test sets.
- Logs hyperparameter results in a CSV file.

---

### Data Handling: `data.py`

Manages trial data, including loading, saving, splitting, and visualizing.

**Classes:**
- `Trial`: Represents a single gesture trial.
- `Data`: Manages a collection of trials.

**Example:**
```python
from data import Data
data = Data("gestures.json")
data.plot()
```

---

## Installation

Install required dependencies:
```bash
pip install tensorflow numpy matplotlib scikit-learn scikit-optimize
```

---
