# Predict the value of BMI(Body Mass Index) from the Gender, Height, and Weight of a person using an MLP with a 3-5-1 topology (3 data inputs, 5 middle neurons, and 1output neuron). 
# Use the bmi.csv dataset for training the MLP. Please don’t use any Python MLP library; instead, use the MLP learning steps outlined in this lecture slide. 
# Use the sigmoid activation function for the MLP and modify your dataset (convert all the data values in the CSV file into floating-point form, 
# for example, 0.179 for  Height 179cm, 0.78 for Weight 78kg, and 0.3 for BMI 3. 
# Similarly, convert the gender strings, such as 0.1 for “Male” and 0.2 for “Female”, or adjust them as needed.
 
# FYI: There is no problem in reducing the size of the bmi.csv file from 500 to 100 for this test.

import math
import pandas as pd

alpha = 0.1

w11 = 0.2; w12 = 0.3; w13 = 0.1; w14 = 0.4; w15 = 0.2
w21 = 0.1; w22 = 0.3; w23 = 0.5; w24 = -0.2; w25 = 0.3  
w31 = 0.4; w32 = 0.1; w33 = 0.4; w34 = 0.3; w35 = 0.1

w41 = -0.3; w42 = 0.4; w43 = -0.1; w44 = 0.3; w45 = 0.2

bias1 = 0.1; bias2 = -0.2; bias3 = 0.3; bias4 = 0.1; bias5 = -0.1
bias6 = 0.2

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def activation_derivative(x):
    return x * (1 - x)

def load_bmi_data(num_samples=100):
    training_data = []
    
    with open('bmi.csv', 'r') as file:
        lines = file.readlines()
        
        for i, line in enumerate(lines[:num_samples]):
            parts = line.strip().split(',')
            gender = parts[0]
            height = int(parts[1])
            weight = int(parts[2])
            
            bmi = weight / ((height / 100) ** 2)
            
            training_data.append((gender, height, weight, round(bmi, 2)))
    
    return training_data

def preprocess_data(gender, height, weight, bmi=None):
    gender_val = 0.1 if gender.lower() == 'male' else 0.2
    height_val = height / 1000.0
    weight_val = weight / 100.0
    bmi_val = bmi / 100.0 if bmi is not None else None
    
    return [gender_val, height_val, weight_val], bmi_val

def forward_propagation(inputs):
    global X1, X2, X3, X4, X5, X6, y1, y2, y3, y4, y5, y6
    
    x1, x2, x3 = inputs[0], inputs[1], inputs[2]
    
    print(f"Forward Pass - Input: x1={x1:.3f}, x2={x2:.3f}, x3={x3:.3f}")
    
    # Hidden Layer Calculations (neurons 1-5)
    X1 = ((x1 * w11) + (x2 * w21) + (x3 * w31)) - bias1
    y1 = sigmoid(X1)  
    print(f"X1 = ((x1*w11) + (x2*w21) + (x3*w31)) - bias1 = {X1:.4f}")
    print(f"y1 = sigmoid(X1) = {y1:.4f}")
    
    X2 = ((x1 * w12) + (x2 * w22) + (x3 * w32)) - bias2
    y2 = sigmoid(X2)
    print(f"X2 = ((x1*w12) + (x2*w22) + (x3*w32)) - bias2 = {X2:.4f}")
    print(f"y2 = sigmoid(X2) = {y2:.4f}")
    
    X3 = ((x1 * w13) + (x2 * w23) + (x3 * w33)) - bias3
    y3 = sigmoid(X3)
    print(f"X3 = ((x1*w13) + (x2*w23) + (x3*w33)) - bias3 = {X3:.4f}")
    print(f"y3 = sigmoid(X3) = {y3:.4f}")
    
    X4 = ((x1 * w14) + (x2 * w24) + (x3 * w34)) - bias4
    y4 = sigmoid(X4)
    print(f"X4 = ((x1*w14) + (x2*w24) + (x3*w34)) - bias4 = {X4:.4f}")
    print(f"y4 = sigmoid(X4) = {y4:.4f}")
    
    X5 = ((x1 * w15) + (x2 * w25) + (x3 * w35)) - bias5
    y5 = sigmoid(X5)
    print(f"X5 = ((x1*w15) + (x2*w25) + (x3*w35)) - bias5 = {X5:.4f}")
    print(f"y5 = sigmoid(X5) = {y5:.4f}")
    
    # Output Layer Calculation
    print("\nOutput Layer Calculation:")
    X6 = ((y1 * w41) + (y2 * w42) + (y3 * w43) + (y4 * w44) + (y5 * w45)) - bias6
    y6 = sigmoid(X6)
    print(f"X6 = ((y1*w41) + (y2*w42) + (y3*w43) + (y4*w44) + (y5*w45)) - bias6 = {X6:.4f}")
    print(f"y6[i] = sigmoid(X6) = {y6:.4f}")
    
    return y6

def calculate_error(target):
    global y6
    
    # e[i] = yd[i] - y6[i] 
    ei = target - y6 
    return ei

def backward_propagation(inputs, target):
    global w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45
    global bias1, bias2, bias3, bias4, bias5, bias6
    global X1, X2, X3, y1, y2, y3, y4, y5, y6
    
    x1, x2, x3 = inputs[0], inputs[1], inputs[2]
    
    print(f"\nBackward Propagation:")
    
    # Calculate error
    ei = target - y6
    
    # EG6 = -yd[i] * (1-y6[i]) * e[i] 
    EG6 = -target * activation_derivative(y6) * ei
    print(f"EG6 = -yd[i] * activation_derivative(y6) * e[i] = {EG6:.6f}")
    
    # Calculate gradients for hidden layer neurons
    EG1 = activation_derivative(y1) * EG6 * w41
    EG2 = activation_derivative(y2) * EG6 * w42  
    EG3 = activation_derivative(y3) * EG6 * w43
    EG4 = activation_derivative(y4) * EG6 * w44
    EG5 = activation_derivative(y5) * EG6 * w45
    
    print(f"EG1 = activation_derivative(y1) * EG6 * w41 = {EG1:.6f}")
    print(f"EG2 = activation_derivative(y2) * EG6 * w42 = {EG2:.6f}")
    print(f"EG3 = activation_derivative(y3) * EG6 * w43 = {EG3:.6f}")
    print(f"EG4 = activation_derivative(y4) * EG6 * w44 = {EG4:.6f}")
    print(f"EG5 = activation_derivative(y5) * EG6 * w45 = {EG5:.6f}")
    
    print("\nWeight Gradient Calculations:")
    
    # Calculate weight gradients for output layer
    Dw41 = alpha * y1 * EG6
    Dw42 = alpha * y2 * EG6
    Dw43 = alpha * y3 * EG6
    Dw44 = alpha * y4 * EG6
    Dw45 = alpha * y5 * EG6
    
    print(f"Dw41 = alpha * y1 * EG6 = {Dw41:.6f}")
    print(f"Dw42 = alpha * y2 * EG6 = {Dw42:.6f}")
    print(f"Dw43 = alpha * y3 * EG6 = {Dw43:.6f}")
    print(f"Dw44 = alpha * y4 * EG6 = {Dw44:.6f}")
    print(f"Dw45 = alpha * y5 * EG6 = {Dw45:.6f}")
    
    # Calculate weight gradients for hidden layer
    Dw11 = alpha * X1 * EG1
    Dw12 = alpha * X2 * EG1
    Dw13 = alpha * X3 * EG1
    
    Dw21 = alpha * X1 * EG2
    Dw22 = alpha * X2 * EG2
    Dw23 = alpha * X3 * EG2
    
    Dw31 = alpha * X1 * EG3
    Dw32 = alpha * X2 * EG3
    Dw33 = alpha * X3 * EG3
    
    Dw14 = alpha * X1 * EG4
    Dw24 = alpha * X2 * EG4
    Dw34 = alpha * X3 * EG4
    
    Dw15 = alpha * X1 * EG5
    Dw25 = alpha * X2 * EG5
    Dw35 = alpha * X3 * EG5
    
    print(f"Hidden Layer Weight Gradients:")
    print(f"  Dw11={Dw11:.6f}, Dw12={Dw12:.6f}, Dw13={Dw13:.6f}")
    print(f"  Dw21={Dw21:.6f}, Dw22={Dw22:.6f}, Dw23={Dw23:.6f}")
    print(f"  Dw31={Dw31:.6f}, Dw32={Dw32:.6f}, Dw33={Dw33:.6f}")
    print(f"  Dw14={Dw14:.6f}, Dw24={Dw24:.6f}, Dw34={Dw34:.6f}")
    print(f"  Dw15={Dw15:.6f}, Dw25={Dw25:.6f}, Dw35={Dw35:.6f}")
    
    # Calculate bias gradients
    Dbias6 = alpha * EG6
    Dbias1 = alpha * EG1
    Dbias2 = alpha * EG2
    Dbias3 = alpha * EG3
    Dbias4 = alpha * EG4
    Dbias5 = alpha * EG5
    
    print(f"\nBias Gradients:")
    print(f"  Dbias1={Dbias1:.6f}, Dbias2={Dbias2:.6f}, Dbias3={Dbias3:.6f}")
    print(f"  Dbias4={Dbias4:.6f}, Dbias5={Dbias5:.6f}, Dbias6={Dbias6:.6f}")
    
    return (Dw11, Dw12, Dw13, Dw14, Dw15, Dw21, Dw22, Dw23, Dw24, Dw25, 
            Dw31, Dw32, Dw33, Dw34, Dw35, Dw41, Dw42, Dw43, Dw44, Dw45,
            Dbias1, Dbias2, Dbias3, Dbias4, Dbias5, Dbias6)

def update_weights(gradients):
    global w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35
    global w41, w42, w43, w44, w45, bias1, bias2, bias3, bias4, bias5, bias6
    
    (Dw11, Dw12, Dw13, Dw14, Dw15, Dw21, Dw22, Dw23, Dw24, Dw25,
     Dw31, Dw32, Dw33, Dw34, Dw35, Dw41, Dw42, Dw43, Dw44, Dw45,
     Dbias1, Dbias2, Dbias3, Dbias4, Dbias5, Dbias6) = gradients
    
    print(f"\nWeight Updates:")
    
    # Update weights
    w11_old = w11; w11 = w11 + Dw11
    w12_old = w12; w12 = w12 + Dw12
    w13_old = w13; w13 = w13 + Dw13
    w14_old = w14; w14 = w14 + Dw14
    w15_old = w15; w15 = w15 + Dw15
    
    w21_old = w21; w21 = w21 + Dw21
    w22_old = w22; w22 = w22 + Dw22
    w23_old = w23; w23 = w23 + Dw23
    w24_old = w24; w24 = w24 + Dw24
    w25_old = w25; w25 = w25 + Dw25
    
    w31_old = w31; w31 = w31 + Dw31
    w32_old = w32; w32 = w32 + Dw32
    w33_old = w33; w33 = w33 + Dw33
    w34_old = w34; w34 = w34 + Dw34
    w35_old = w35; w35 = w35 + Dw35
    
    # Update weights
    w41_old = w41; w41 = w41 + Dw41
    w42_old = w42; w42 = w42 + Dw42
    w43_old = w43; w43 = w43 + Dw43
    w44_old = w44; w44 = w44 + Dw44
    w45_old = w45; w45 = w45 + Dw45
    
    # Update biases
    bias1_old = bias1; bias1 = bias1 + Dbias1
    bias2_old = bias2; bias2 = bias2 + Dbias2
    bias3_old = bias3; bias3 = bias3 + Dbias3
    bias4_old = bias4; bias4 = bias4 + Dbias4
    bias5_old = bias5; bias5 = bias5 + Dbias5
    bias6_old = bias6; bias6 = bias6 + Dbias6
    
    print(f"Hidden Layer 1: w11: {w11_old:.4f} → {w11:.4f}")
    print(f"Hidden Layer 2: w21: {w21_old:.4f} → {w21:.4f}")
    print(f"Hidden Layer 3: w31: {w31_old:.4f} → {w31:.4f}")
    print(f"Hidden Layer 4: w14: {w14_old:.4f} → {w14:.4f}")
    print(f"Hidden Layer 5: w15: {w15_old:.4f} → {w15:.4f}")
    print(f"Output Layer: w41: {w41_old:.4f} → {w41:.4f}")
    print(f"Biases: bias1: {bias1_old:.4f} → {bias1:.4f}, bias6: {bias6_old:.4f} → {bias6:.4f}")

def train_single_sample(gender, height, weight, target_bmi):
    print(f"TRAINING SAMPLE: {gender}, {height}cm, {weight}kg, BMI={target_bmi}")
    
    # Preprocess data
    inputs, target = preprocess_data(gender, height, weight, target_bmi)
    print(f"Preprocessed - Input: {inputs}, Target: {target:.4f}")
    
    # Forward propagation
    output_val = forward_propagation(inputs)
    
    # Calculate error
    error = calculate_error(target)
    
    # Backward propagation
    gradients = backward_propagation(inputs, target)
    
    # Update weights
    update_weights(gradients)
    
    return error

def predict(gender, height, weight, show_details=False):
    inputs, _ = preprocess_data(gender, height, weight)
    
    if show_details:
        print(f"\nPREDICTION for {gender}, {height}cm, {weight}kg:")
        print("-" * 50)
        output_val = forward_propagation(inputs)
    else:
        global X1, X2, X3, y1, y2, y3, y4, y5, y6
        
        x1, x2, x3 = inputs[0], inputs[1], inputs[2]
        
        X1 = ((x1 * w11) + (x2 * w12) + (x3 * w13)) - bias1
        y1 = sigmoid(X1)
        X2 = ((x1 * w21) + (x2 * w22) + (x3 * w23)) - bias2
        y2 = sigmoid(X2)
        X3 = ((x1 * w31) + (x2 * w32) + (x3 * w33)) - bias3
        y3 = sigmoid(X3)
        X4 = ((x1 * w14) + (x2 * w24) + (x3 * w34)) - bias4
        y4 = sigmoid(X4)
        X5 = ((x1 * w15) + (x2 * w25) + (x3 * w35)) - bias5
        y5 = sigmoid(X5)
        X6 = ((y1 * w41) + (y2 * w42) + (y3 * w43) + (y4 * w44) + (y5 * w45)) - bias6
        y6 = sigmoid(X6)
    
    predicted_bmi = y6 * 100
    return predicted_bmi

def manual_calculation_example():
    print("MLP CALCULATION EXAMPLE")
    print("="*50)
    
    gender = "Male"
    height = 175  # cm
    weight = 70   # kg
    target_bmi = 22.86
    
    print("Initial Weights (3-5-1 topology):")
    print(f"Input to Hidden: w11={w11}, w12={w12}, w13={w13}")
    print(f"                 w21={w21}, w22={w22}, w23={w23}")
    print(f"                 w31={w31}, w32={w32}, w33={w33}")
    print(f"                 w14={w14}, w24={w24}, w34={w34}")
    print(f"                 w15={w15}, w25={w25}, w35={w35}")
    print(f"Hidden to Output: w41={w41}, w42={w42}, w43={w43}, w44={w44}, w45={w45}")
    
    error = train_single_sample(gender, height, weight, target_bmi)
    
    print(f"\nFinal Error: {error:.6f}")

def training_example():
    print("\n\nTRAINING EXAMPLE WITH BMI DATASET")
    print("="*60)
    
    training_data = load_bmi_data(100)
    
    print(f"Loaded {len(training_data)} samples from BMI dataset")
    print("Sample data:")
    for i in range(5):  # Show first 5 samples
        gender, height, weight, bmi = training_data[i]
        print(f"  {gender}, {height}cm, {weight}kg, BMI={bmi}")
    print("...")
    
    print(f"\nTraining on {len(training_data)} samples for 3 epochs:")
    
    for epoch in range(3):
        print(f"\nEPOCH {epoch + 1}")
        print("-" * 40)
        total_error = 0
        
        # Train on all samples (but only show detailed output for first few)
        for i, (gender, height, weight, target_bmi) in enumerate(training_data):
            if i < 3:  # Show detailed output for first 3 samples only
                print(f"\nSample {i+1}/{len(training_data)}")
                error = train_single_sample(gender, height, weight, target_bmi)
            else:
                # Train silently for remaining samples
                inputs, target = preprocess_data(gender, height, weight, target_bmi)
                output_val = forward_propagation_silent(inputs)
                error = target - y6
                gradients = backward_propagation_silent(inputs, target)
                update_weights_silent(gradients)
            
            total_error += abs(error)
        
        avg_error = total_error / len(training_data)
        print(f"\nEpoch {epoch + 1} Average Error: {avg_error:.6f}")
    
    # Test predictions on first 10 samples
    print("\n\nTEST PREDICTIONS (First 10 samples):")
    print("-" * 50)
    for i in range(10):
        gender, height, weight, actual_bmi = training_data[i]
        predicted_bmi = predict(gender, height, weight)
        error = abs(actual_bmi - predicted_bmi)
        print(f"{gender:6s} {height:3d}cm {weight:2d}kg | Actual: {actual_bmi:5.2f} | Predicted: {predicted_bmi:5.2f} | Error: {error:.2f}")

def forward_propagation_silent(inputs):
    """Silent forward propagation for batch training"""
    global X1, X2, X3, X4, X5, X6, y1, y2, y3, y4, y5, y6
    
    x1, x2, x3 = inputs[0], inputs[1], inputs[2]
    
    X1 = ((x1 * w11) + (x2 * w21) + (x3 * w31)) - bias1
    y1 = sigmoid(X1)
    X2 = ((x1 * w12) + (x2 * w22) + (x3 * w32)) - bias2
    y2 = sigmoid(X2)
    X3 = ((x1 * w13) + (x2 * w23) + (x3 * w33)) - bias3
    y3 = sigmoid(X3)
    X4 = ((x1 * w14) + (x2 * w24) + (x3 * w34)) - bias4
    y4 = sigmoid(X4)
    X5 = ((x1 * w15) + (x2 * w25) + (x3 * w35)) - bias5
    y5 = sigmoid(X5)
    X6 = ((y1 * w41) + (y2 * w42) + (y3 * w43) + (y4 * w44) + (y5 * w45)) - bias6
    y6 = sigmoid(X6)
    
    return y6

def backward_propagation_silent(inputs, target):
    """Silent backward propagation for batch training"""
    global w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35, w41, w42, w43, w44, w45
    global y1, y2, y3, y4, y5, y6
    
    x1, x2, x3 = inputs[0], inputs[1], inputs[2]
    ei = target - y6
    
    # Output layer gradients
    EG6 = -target * activation_derivative(y6) * ei
    
    # Hidden layer gradients
    EG1 = activation_derivative(y1) * EG6 * w41
    EG2 = activation_derivative(y2) * EG6 * w42
    EG3 = activation_derivative(y3) * EG6 * w43
    EG4 = activation_derivative(y4) * EG6 * w44
    EG5 = activation_derivative(y5) * EG6 * w45
    
    # Weight gradients for output layer
    Dw41 = alpha * y1 * EG6
    Dw42 = alpha * y2 * EG6
    Dw43 = alpha * y3 * EG6
    Dw44 = alpha * y4 * EG6
    Dw45 = alpha * y5 * EG6
    
    # Weight gradients for hidden layer
    Dw11 = alpha * x1 * EG1; Dw12 = alpha * x2 * EG1; Dw13 = alpha * x3 * EG1
    Dw21 = alpha * x1 * EG2; Dw22 = alpha * x2 * EG2; Dw23 = alpha * x3 * EG2
    Dw31 = alpha * x1 * EG3; Dw32 = alpha * x2 * EG3; Dw33 = alpha * x3 * EG3
    Dw14 = alpha * x1 * EG4; Dw24 = alpha * x2 * EG4; Dw34 = alpha * x3 * EG4
    Dw15 = alpha * x1 * EG5; Dw25 = alpha * x2 * EG5; Dw35 = alpha * x3 * EG5
    
    # Bias gradients
    Dbias1 = alpha * EG1; Dbias2 = alpha * EG2; Dbias3 = alpha * EG3
    Dbias4 = alpha * EG4; Dbias5 = alpha * EG5; Dbias6 = alpha * EG6
    
    return (Dw11, Dw12, Dw13, Dw14, Dw15, Dw21, Dw22, Dw23, Dw24, Dw25,
            Dw31, Dw32, Dw33, Dw34, Dw35, Dw41, Dw42, Dw43, Dw44, Dw45,
            Dbias1, Dbias2, Dbias3, Dbias4, Dbias5, Dbias6)

def update_weights_silent(gradients):
    global w11, w12, w13, w14, w15, w21, w22, w23, w24, w25, w31, w32, w33, w34, w35
    global w41, w42, w43, w44, w45, bias1, bias2, bias3, bias4, bias5, bias6
    
    (Dw11, Dw12, Dw13, Dw14, Dw15, Dw21, Dw22, Dw23, Dw24, Dw25,
     Dw31, Dw32, Dw33, Dw34, Dw35, Dw41, Dw42, Dw43, Dw44, Dw45,
     Dbias1, Dbias2, Dbias3, Dbias4, Dbias5, Dbias6) = gradients
    
    # Update all weights
    w11 += Dw11; w12 += Dw12; w13 += Dw13; w14 += Dw14; w15 += Dw15
    w21 += Dw21; w22 += Dw22; w23 += Dw23; w24 += Dw24; w25 += Dw25
    w31 += Dw31; w32 += Dw32; w33 += Dw33; w34 += Dw34; w35 += Dw35
    w41 += Dw41; w42 += Dw42; w43 += Dw43; w44 += Dw44; w45 += Dw45
    
    # Update all biases
    bias1 += Dbias1; bias2 += Dbias2; bias3 += Dbias3
    bias4 += Dbias4; bias5 += Dbias5; bias6 += Dbias6

manual_calculation_example()

training_example()

