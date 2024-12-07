import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = './Lightweight_results.csv'
data = pd.read_csv(file_path)

# Define the three conditions
condition_both_zero = (data['r_edge_count'] == 0) & (data['b_edge_count'] == 0)
condition_one_zero = ((data['r_edge_count'] == 0) & (data['b_edge_count'] > 0)) | \
                     ((data['b_edge_count'] == 0) & (data['r_edge_count'] > 0))
condition_both_non_zero = (data['r_edge_count'] > 0) & (data['b_edge_count'] > 0)

# Calculate success rate for each condition
success_both_zero = data[condition_both_zero]['success'].mean()
success_one_zero = data[condition_one_zero]['success'].mean()
success_both_non_zero = data[condition_both_non_zero]['success'].mean()

# Prepare data for visualization
categories = ['Both Zero', 'One Zero', 'Both Non-Zero']
success_rates = [success_both_zero, success_one_zero, success_both_non_zero]

# Plot the success rates
plt.figure(figsize=(8, 6))
plt.bar(categories, success_rates, color=['blue', 'orange', 'green'], edgecolor='black')
plt.title('Success Rate by Edge Count Conditions')
plt.ylabel('Success Rate')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
