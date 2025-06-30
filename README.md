
**Curse of Dimensionality and Classifier Performance
**

This project explores the **curse of dimensionality** and evaluates the performance of various classifiers on high-dimensional data using `scikit-learn`. It includes simulations of distance metrics in high dimensions and compares classification models on a custom dataset.

## Course Info

* **Course**: Artificial Intelligence
* **Instructor**: Professor Ansaf Sales-Aouissi
* **Institution**: Columbia University

## Project Structure

```text
├── curse.py               # Simulation and visualization of the curse of dimensionality
├── classifiers.py         # Classifier management and evaluation
├── output.csv             # Dataset used for classification tasks
```

## Project Components

### `curse.py` – Curse of Dimensionality

This script visualizes how the ratio between maximum and minimum distances (`d_max/d_min`) changes with increasing feature dimensions for different sample sizes. It:

* Generates synthetic datasets using `make_classification`
* Computes pairwise distances in each generated dataset
* Logs the ratio of max/min distances
* Plots log-scaled ratios against number of features to illustrate the curse of dimensionality

> Output: A matplotlib plot showing `log(d_max/d_min)` vs. feature dimensions for sample sizes {100, 200, 500, 1000}

---

### `classifiers.py` – Classifier Comparison

Defines a `ClassifierManager` class that:

* Preprocesses data from `output.csv`
* Splits it into training and testing sets
* Trains and evaluates multiple classifiers:

  * K-Nearest Neighbors
  * Logistic Regression
  * Decision Tree
  * Random Forest
  * AdaBoost
  * Support Vector Machine
* Optionally performs GridSearchCV for model tuning
* Outputs and visualizes classification results and accuracies

---

##  Dataset

**File**: `output.csv`

* Contains labeled data used for training and testing classifiers
* Format: Features in all columns except the last, which contains labels

##  Dependencies

```bash
numpy
pandas
matplotlib
scikit-learn
```

Install via pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```

##  How to Run

1. Run `curse.py` to visualize effects of increasing feature dimensions:

```bash
python curse.py
```

2. Use `classifiers.py` by instantiating the `ClassifierManager` with a DataFrame loaded from `output.csv`. Then, train and evaluate classifiers as needed.

Example usage (within a script or Jupyter notebook):

```python
import pandas as pd
from classifiers import ClassifierManager

data = pd.read_csv("output.csv")
manager = ClassifierManager(data)
manager.evaluate_all_classifiers()  # (Assumes this method exists for bulk evaluation)
```

---

##  Acknowledgments

* Professor **Ansaf Sales-Aouissi**
* Columbia University AI Department


