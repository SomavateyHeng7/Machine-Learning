#logistic regression with bmi.csv

import pandas as pd

# Load dataset
bmi = pd.read_csv('bmi.csv', names=['Gender', 'Height', 'Weight', 'BMI_Class'])
print(bmi)

# calculate sums and means
sum_height = bmi['Height'].sum()
sum_weight = bmi['Weight'].sum()
sum_bmi = bmi['BMI_Class'].sum()

mean_height = bmi['Height'].mean()
mean_weight = bmi['Weight'].mean()
mean_bmi = bmi['BMI_Class'].mean()

#calculate the square sum
bmi['Height_Squared'] = bmi['Height'] ** 2
bmi['Weight_Squared'] = bmi['Weight'] ** 2
bmi['Height_Weight'] = bmi['Height'] * bmi['Weight']
bmi['BMI_Height'] = bmi['BMI_Class'] * bmi['Height']
bmi['BMI_Weight'] = bmi['BMI_Class'] * bmi['Weight']

sum_height_squared = bmi['Height_Squared'].sum()             
sum_weight_squared = bmi['Weight_Squared'].sum()             
sum_height_weight = bmi['Height_Weight'].sum()               
sum_bmi_height = bmi['BMI_Height'].sum()                     
sum_bmi_weight = bmi['BMI_Weight'].sum()                     

#calculate regression sums
n = len(bmi) 

reg_sum_height_squared = sum_height_squared - (sum_height ** 2) / n
reg_sum_weight_squared = sum_weight_squared - (sum_weight ** 2) / n
reg_sum_height_bmi = sum_bmi_height - (sum_height * sum_bmi) / n
reg_sum_weight_bmi = sum_bmi_weight - (sum_weight * sum_bmi) / n
reg_sum_height_weight = sum_height_weight - (sum_height * sum_weight) / n

#calculate B0, B1, B2 
denominator = (reg_sum_height_squared * reg_sum_weight_squared) - (reg_sum_height_weight ** 2)

B1_numerator = (reg_sum_weight_squared * reg_sum_height_bmi) - (reg_sum_height_weight * reg_sum_weight_bmi)
B1 = B1_numerator / denominator
print(f"B1: {B1}")

B2_numerator = (reg_sum_height_squared * reg_sum_weight_bmi) - (reg_sum_height_weight * reg_sum_height_bmi)
B2 = B2_numerator / denominator
print(f"B2: {B2}")

B0 = mean_bmi - B1 * mean_height - B2 * mean_weight
print(f"B0: {B0}")

# Estimate BMI using regression model
estimated_bmi = B0 + B1 * mean_height + B2 * mean_weight
print(f"Estimated BMI: {estimated_bmi:.3f}")

# BMI categories
bmi_categories = [
    "Extremely Weak",
    "Weak",
    "Normal",
    "Overweight",
    "Obesity",
    "Extreme Obesity"
]

def predict_index(height, weight):
    raw_bmi = B0 + B1 * height + B2 * weight
    rounded_bmi = int(round(raw_bmi))
    rounded_bmi = max(0, min(rounded_bmi, len(bmi_categories) - 1))
    category = bmi_categories[rounded_bmi]
    print(f"Raw index: {raw_bmi:.3f}")
    print(f"Rounded index: {rounded_bmi}")
    print(f"Category: {category}")


height = int(input("Enter height: "))
weight = int(input("Enter weight: "))
predict_index(height, weight)

