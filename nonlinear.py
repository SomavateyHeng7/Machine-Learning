# Consider a perceptron with two real-valued inputs and an output unit
# with a sigmoid activation function (for non-linearly separable data).
# All the initial weights and the bias are equal to 0.5. Assume that the
# output (y) should be 0.6 for the input x1 = 0.3 and x2 = 0.4. Show
# how the iterations on the given data supports the perceptron to
# reach the output (assume α = 0.1)

import numpy as np

x1 = 0.3
x2 = 0.4
yd = 0.6
w1 = 0.5
w2 = 0.5  
b = 0.5
alpha = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def perceptron(x1, x2, w1, w2, b):
    z = w1 * x1 + w2 * x2 - b
    y_pred = sigmoid(z)
    return z, y_pred

def calculate_gradients(x1, x2, y_target, y_pred):
    error = y_target - y_pred
    delta = error * y_pred * (1 - y_pred)
    
    grad_w1 = delta * x1
    grad_w2 = delta * x2
    grad_b = -delta  
    
    return grad_w1, grad_w2, grad_b, error, delta

def update_weights(w1, w2, b, grad_w1, grad_w2, grad_b, alpha):
    w1_new = w1 + alpha * grad_w1
    w2_new = w2 + alpha * grad_w2
    b_new = b + alpha * grad_b
    
    return w1_new, w2_new, b_new

def train_perceptron():
    global w1, w2, b
    
    print("Iteration | x1  | x2  | Yd  | w1     | w2     | Y      | e      | w1_new | w2_new")
    print("-" * 85)
    
    iteration = 1
    
    while True:
        # Store weights before update for display
        w1_before = w1
        w2_before = w2
        
        # perceptron
        z, y_pred = perceptron(x1, x2, w1, w2, b)

        # Calculate gradients
        grad_w1, grad_w2, grad_b, error, delta = calculate_gradients(x1, x2, yd, y_pred)
        
        # Update weights
        w1, w2, b = update_weights(w1, w2, b, grad_w1, grad_w2, grad_b, alpha)
        
        print(f"    {iteration:2d}    | {x1} | {x2} | {yd} | {w1_before} | {w2_before} | {y_pred} | {error} | {w1} | {w2}")
        
        if abs(error) < 0.01:
            break
            
        iteration += 1
    
    print()
    
train_perceptron()
