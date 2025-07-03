# Optimization Library

A Python library for solving linear, nonlinear, and integer programming problems. The library provides a collection of optimization algorithms for tasks such as the diet problem, model parameter optimization, and the 0-1 knapsack problem.

## Features

- **Linear Programming**: Solve problems like the diet problem using methods such as Simplex, Relaxation, Column Generation, ADMM, and Mirror Descent.
- **Nonlinear Programming**: Optimize model parameters with methods like Gradient Descent, Newton's Method, Steepest Descent, Adam, and Nelder-Mead.
- **Integer Programming**: Solve the 0-1 knapsack problem using Branch and Bound, Gomory Cuts, Cutting Planes, Lagrangian Relaxation, and Sherali-Adams (Level 1).
- Visualization and logging of optimization results using Matplotlib and Pandas.

## Installation

Install the library using pip:

```bash
pip install optimization-library
```

Requirements
- numpy~=2.3.0
- cvxpy~=1.6.5
- pandas~=2.3.0
- matplotlib~=3.10.3
- PuLP~=3.2.1
- scipy~=1.15.3

# Usage
Linear Programming
```bash
import numpy as np
from optimization_library import solve_lp, post_processing_linear_approximation_logs

c = np.array([3, 2], dtype=float)
A = np.array([[1, 1], [2, 1]], dtype=float)
b = np.array([4, 5], dtype=float)
methods = ["simplex", "ADMM"]

results = [solve_lp(method, c, A, b, epsilon=1e-6, is_maximization=True) for method in methods]
post_processing_linear_approximation_logs(results, visual=True, file_print=True)
```
Nonlinear Programming
```bash
import numpy as np
from optimization_library import solve_nlp, post_processing_non_linear_approximation_logs

t = np.linspace(0, 2 * np.pi, 10)
y = np.exp(-0.5 * t) + np.sin(-1.2 * t)
model = lambda x, t: np.exp(-x[0] * t) + np.sin(x[1] * t)
x0 = [0.25, 0.25]
bounds = [(-3.5, 2.5), (-4.2, 2.8)]
methods = ["adam", "nelder-mead"]

results = [solve_nlp(method, x0, t, y, model=model, bounds=bounds, epsilon=1e-6) for method in methods]
post_processing_non_linear_approximation_logs(results, visual=True, file_print=True)
```
Integer Programming
```bash
from optimization_library import solve_ip, post_processing_integer_approximation_logs

weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 10
methods = ["branch_and_bound", "gomory"]

results = [solve_ip(method, weights, values, capacity, epsilon=1e-4) for method in methods]
post_processing_integer_approximation_logs(results, visual=True, file_print=True)
```

# Examples
A console application demonstrating the usage of the library is available in the GitHub repository under the examples/ directory. Run it with:
```bash
git clone https://github.com/UWFms/optimization-library.git
cd optimization-library/examples
python main.py
```
# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Contributing
Contributions are welcome! Please submit issues or pull requests to the GitHub repository https://github.com/UWFms/optimization-library.

# Documentation
Full documentation is available at https://disk.yandex.ru/i/k7wqcUCCx1Ym9Q.