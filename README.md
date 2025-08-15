# Machine Learning Algorithms Implementation

This repository contains implementations of fundamental machine learning algorithms and neural networks from scratch, without using high-level ML libraries. All algorithms are implemented in Python with detailed step-by-step calculations for educational purposes.

## 📁 Project Structure

```
├── bmi.csv                    # BMI dataset (500 samples)
├── geneticalgorithm.py        # Genetic Algorithm optimization
├── logisticregression.py      # Multiple Linear Regression for BMI prediction
├── mlp_bmi.py                # Multi-Layer Perceptron (3-5-1 topology)
├── nonlinear.py              # Non-linear Perceptron with sigmoid activation
├── perceptron.py             # Basic Perceptron for OR/AND functions
└── README.md                 # This file
```

## 🧠 Algorithms Implemented

### 1. Perceptron (`perceptron.py`)
- **Purpose**: Implements a basic perceptron for linearly separable data
- **Functions**: OR and AND logic gates
- **Features**:
  - Step activation function
  - Training with error correction
  - Weight updates using gradient descent
- **Initial Parameters**: w1=0.3, w2=-0.1, b=0.2, α=0.1

### 2. Non-linear Perceptron (`nonlinear.py`)
- **Purpose**: Handles non-linearly separable data using sigmoid activation
- **Features**:
  - Sigmoid activation function
  - Gradient-based weight updates
  - Convergence to target output
- **Example**: Converges to output 0.6 for inputs x1=0.3, x2=0.4

### 3. Multi-Layer Perceptron (`mlp_bmi.py`)
- **Purpose**: BMI prediction using a 3-layer neural network
- **Architecture**: 3-5-1 topology (3 inputs, 5 hidden neurons, 1 output)
- **Features**:
  - Forward and backward propagation
  - Sigmoid activation function
  - Data preprocessing (normalization)
  - Training on BMI dataset
- **Inputs**: Gender, Height, Weight → **Output**: BMI category

### 4. Multiple Linear Regression (`logisticregression.py`)
- **Purpose**: BMI prediction using statistical regression
- **Method**: Least squares regression with multiple variables
- **Features**:
  - Manual calculation of regression coefficients (B0, B1, B2)
  - BMI category classification
  - Interactive prediction interface
- **Formula**: BMI = B0 + B1×Height + B2×Weight

### 5. Genetic Algorithm (`geneticalgorithm.py`)
- **Purpose**: Optimization using evolutionary computation
- **Problem**: Find values for equation a + 2b + 3c + 4d = 30
- **Features**:
  - Population-based search
  - Crossover and mutation operations
  - Fitness-based selection
  - Convergence tracking

## 📊 Dataset

### BMI Dataset (`bmi.csv`)
- **Size**: 500 samples
- **Format**: Gender, Height(cm), Weight(kg), BMI_Class
- **Classes**: 
  - 0: Extremely Weak
  - 1: Weak  
  - 2: Normal
  - 3: Overweight
  - 4: Obesity
  - 5: Extreme Obesity

**Sample Data:**
```
Male,174,96,4
Female,185,110,4
Male,189,87,2
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy
```

### Running the Code

1. **Perceptron Training:**
```bash
python perceptron.py
# Choose: 1 for OR function, 2 for AND function
```

2. **Non-linear Perceptron:**
```bash
python nonlinear.py
# Shows convergence iterations for sigmoid perceptron
```

3. **BMI Prediction with MLP:**
```bash
python mlp_bmi.py
# Trains neural network and shows detailed calculations
```

4. **BMI Regression Analysis:**
```bash
python logisticregression.py
# Enter height and weight for BMI prediction
```

5. **Genetic Algorithm Optimization:**
```bash
python geneticalgorithm.py
# Finds optimal solution for target equation
```

## 🔬 Key Features

### Educational Focus
- **Step-by-step calculations** shown for each algorithm
- **Manual implementation** without ML libraries
- **Detailed mathematical formulations**
- **Weight update visualizations**

### Algorithm Implementations
- **From-scratch neural networks** with manual forward/backward propagation
- **Custom activation functions** (step, sigmoid)
- **Manual gradient calculations**
- **Statistical regression** using mathematical formulas

### Data Processing
- **Normalization techniques** for neural network inputs
- **Data preprocessing** for different input types
- **Category encoding** for classification tasks

## 📈 Results and Performance

### MLP Performance
- Trains on 100 BMI samples
- 3 epochs with detailed weight updates
- Prediction accuracy shown for test samples

### Genetic Algorithm
- Population size: 5 chromosomes
- Gene range: 1-30
- Convergence typically within 10-50 generations

### Regression Analysis
- Manual coefficient calculation
- Category-based BMI classification
- Interactive prediction system

## 🧮 Mathematical Concepts

### Neural Networks
- **Forward Propagation**: Input → Hidden → Output
- **Backward Propagation**: Error gradient calculation
- **Weight Updates**: W_new = W_old + α × gradient
- **Activation Functions**: Sigmoid, Step function

### Optimization
- **Genetic Operators**: Selection, Crossover, Mutation
- **Fitness Functions**: Inverse error-based selection
- **Population Evolution**: Generational improvement

### Regression
- **Least Squares Method**: Minimizing sum of squared errors
- **Multiple Variable Regression**: BMI = f(height, weight)
- **Coefficient Calculation**: Using matrix mathematics

## 🎯 Learning Objectives

This repository demonstrates:
- **Neural network fundamentals** from basic perceptrons to MLPs
- **Optimization techniques** using evolutionary algorithms
- **Statistical learning** through regression analysis
- **Manual implementation** of ML algorithms
- **Mathematical understanding** of gradient descent and backpropagation

## 📚 Educational Value

Perfect for:
- **Machine Learning students** learning algorithm internals
- **Understanding mathematical foundations** of ML
- **Implementing algorithms from scratch**
- **Comparing different learning approaches**
- **Visualizing algorithm convergence**

## 🔧 Technical Details

### Implementation Notes
- Pure Python implementations
- No external ML libraries (sklearn, tensorflow, etc.)
- Mathematical operations using basic numpy/pandas
- Detailed logging of algorithm steps

### Code Structure
- Modular function design
- Clear variable naming
- Extensive comments and documentation
- Educational print statements

## 🤝 Contributing

Feel free to contribute by:
- Adding more algorithms
- Improving documentation
- Optimizing implementations
- Adding visualization features

## 📄 License

This project is open source.

---

*This repository serves as an educational resource for understanding the mathematical foundations of machine learning algorithms through hands-on implementation.*
