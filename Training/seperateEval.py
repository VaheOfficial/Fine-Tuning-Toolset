import json
import random

# Load the JSON data from file
input_file = 'C:/Users/VaheOfficial/projects/Data Science/Detroit Data/detroit_become_human_alpaca_fine_tuning_data.json'  # Replace with the path to your JSON file
with open(input_file, 'r') as f:
    data = json.load(f)

# Shuffle the data to ensure random distribution
random.shuffle(data)

# Define the split ratio (e.g., 80% for training and 20% for evaluation)
split_ratio = 0.8
split_index = int(len(data) * split_ratio)

# Split the data
train_data = data[:split_index]
eval_data = data[split_index:]

# Save the split datasets into separate JSON files
train_file = 'C:/Users/VaheOfficial/projects/Data Science/Datasets/Training/train_data.json'  # Replace with the desired path for the training data
eval_file = 'C:/Users/VaheOfficial/projects/Data Science/Datasets/Eval/eval_data.json'    # Replace with the desired path for the evaluation data

with open(train_file, 'w') as f:
    json.dump(train_data, f, indent=4)

with open(eval_file, 'w') as f:
    json.dump(eval_data, f, indent=4)

print(f'Training data saved to {train_file}')
print(f'Evaluation data saved to {eval_file}')
