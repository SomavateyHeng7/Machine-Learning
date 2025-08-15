# Train a perceptron for getting 2-input OR and AND functions.
# Assume w1 =0.3, w2 = -0.1, b = 0.2 and ∝ = 0.1


w1 = 0.3
w2 = -0.1
b = 0.2 
alpha = 0.1  

or_func = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1)
]

and_func = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]

choice = input(" Train function 1: OR or 2:AND\n")

if choice == "1":
    training_data = or_func
    function_name = "OR"
elif choice == "2":
    training_data = and_func
    function_name = "AND"
else:
    print("Invalid choice.")
    exit()

def stepactivation(x):
    return 1 if x >= 0 else 0

def cal_perceptron(x1, x2, w1, w2, b):
    z = w1 * x1 + w2 * x2 - b 
    y = stepactivation(z)
    return y, z

def train_perceptron(training_data, w1, w2, b, alpha, max_iterations=100):
    
    print("Epoch | x1 | x2 | Yd | w1  | w2  | Y | e | w1  | w2")
    print("-" * 50)
    
    epoch = 1
    iteration = 0
    max_iter = max_iterations
    
    #epoch loop
    while iteration < max_iter:
        all_correct = True
        epoch_started = False

        for i, (x1, x2, yd) in enumerate(training_data):
            iteration += 1
            y, weighted_sum = cal_perceptron(x1, x2, w1, w2, b)
            error = yd - y
            
            old_w1, old_w2 = w1, w2

            if not epoch_started:
                print(f"  {epoch}   | {x1}  | {x2}  | {yd}  |{old_w1}     |{old_w2}      | {y}     |{error} |", end="")
                epoch_started = True
            else:
                print(f"      | {x1}  | {x2}  | {yd}  |{old_w1}      |{old_w2}      | {y}    |{error} |", end="")

            if error != 0:
                all_correct = False

                delta_w1 = alpha * x1 * error
                delta_w2 = alpha * x2 * error
            
                w1 += delta_w1
                w2 += delta_w2

                print(f"{w1:4.1f} |{w2:4.1f}")
            else:
                print(f"{old_w1:4.1f} |{old_w2:4.1f}")

        if all_correct:
            print()
            break
        
        epoch += 1
        
        if iteration >= max_iter:
            print("Maximum iterations reached.")
            break

    return w1, w2, b

final_w1, final_w2, final_b = train_perceptron(training_data, w1, w2, b, alpha)



