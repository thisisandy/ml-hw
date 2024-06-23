```markdown
# Optimization Algorithms Comparison

This repository contains Jupyter notebooks that demonstrate the application of various optimization algorithms to different problems, including the N-Queen problem, One-Max problem, and neural network optimization.

## Notebooks

### 1. N-Queen Problem

**Notebook**: `N-Queen.ipynb`

This notebook explores the application of randomized optimization algorithms to solve the N-Queen problem, an NP-hard problem where the goal is to place N queens on an N×N chessboard such that no two queens threaten each other.

**Algorithms Covered:**
- Randomized Hill Climbing (RHC)
- Simulated Annealing (SA)
- Genetic Algorithm (GA)
- MIMIC (Estimation of Distribution Algorithm)

**Key Results:**
- Performance comparison of algorithms as N varies.
- Execution time analysis.
- Fitness curve analysis.

### 2. One-Max Problem

**Notebook**: `one-max.ipynb`

This notebook investigates the One-Max problem, a classic benchmark problem for evaluating optimization algorithms. The objective is to maximize the number of 1s in a binary string.

**Algorithms Covered:**
- Randomized Hill Climbing (RHC)
- Simulated Annealing (SA)
- Genetic Algorithm (GA)
- MIMIC (Estimation of Distribution Algorithm)

**Key Results:**
- Best fitness values for varying string lengths.
- Execution time comparison.
- Fitness curve analysis.

### 3. Neural Network Optimization

**Notebook**: `nn.ipynb`

This notebook demonstrates the optimization of a neural network's hyperparameters using various optimization algorithms. The performance of the neural network is evaluated based on accuracy and F1 scores.

**Algorithms Covered:**
- Randomized Hill Climbing (RHC)
- Simulated Annealing (SA)
- Genetic Algorithm (GA)
- Traditional Backpropagation

**Key Results:**
- Comparison of accuracy, F1 score, training time, and prediction time.
- Impact of hyperparameter tuning on performance.
- Learning curve analysis.

## Getting Started

To run the notebooks, you will need to have Python installed along with Jupyter Notebook or Jupyter Lab. The required Python packages are listed in the `requirements.txt` file.

### Prerequisites

- Python 3.12.2 or higher
- Jupyter Notebook or Jupyter Lab

### Installation

1. Clone the repository:
   ```sh
   git clone git@github.com:thisisandy/ml-hw.git
   cd ml-hw
   git checkout hw2
   cd hw2
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook or Jupyter Lab:
   ```sh
   jupyter notebook
   # or
   jupyter lab
   ```

4. Open the desired notebook (`N-Queen.ipynb`, `one-max.ipynb`, or `nn.ipynb`) and run the cells.

## Results Summary

### One-Max Problem
- **RHC and SA**: Efficient in finding fair solutions quickly.
- **GA and MIMIC**: Provided the most accurate results, with GA taking longer per iteration and MIMIC requiring more computational time overall.

### N-Queen Problem
- **RHC and SA**: Often found local optima as problem size increased.
- **GA and MIMIC**: Better performance with MIMIC showing superior efficiency in solution quality and computational time.

### Neural Network Optimization
- **Backpropagation**: Most efficient and effective, consistently achieving high accuracy and F1 scores.
- **GA**: Accurate but time-intensive.
- **RHC and SA**: Lower performance in accuracy and F1 scores, less sensitive to hyperparameter changes.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
```
