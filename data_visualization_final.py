import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = './체급별_예측결과.csv'
data = pd.read_csv(file_path)

# Ensure numeric columns are properly converted
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='ignore')

# Combine Red and Blue previous fights into one column
previous_fights_red = pd.to_numeric(data['r_wins_total'], errors='coerce') + pd.to_numeric(data['r_losses_total'], errors='coerce')
previous_fights_blue = pd.to_numeric(data['b_wins_total'], errors='coerce') + pd.to_numeric(data['b_losses_total'], errors='coerce')
data_combined = pd.DataFrame({
    'previous_fights': pd.concat([previous_fights_red, previous_fights_blue], ignore_index=True),
    'success': pd.concat([data['success'], data['success']], ignore_index=True)
})

# Function to visualize success rate by multiple attributes
def visualize_success_rate(data, success_col, attribute_cols):
    integer_columns = []
    float_columns = []

    # Separate columns into integer and float types
    for column in attribute_cols:
        if np.issubdtype(data[column].dtype, np.integer):
            integer_columns.append(column)
        elif np.issubdtype(data[column].dtype, np.float64) or np.issubdtype(data[column].dtype, np.float32):
            float_columns.append(column)

    # Plot integer columns (bar charts)
    plt.figure(figsize=(15, len(integer_columns) * 6))
    for idx, column in enumerate(integer_columns, 1):
        values = data[column].dropna()
        column_name = column.replace('r_', '').replace('b_', '')  # Remove 'r_' and 'b_' prefixes
        unique_values = sorted(values.unique())
        if len(unique_values) <= 4:
            bins = unique_values + [unique_values[-1] + 1]
        else:
            bins = np.linspace(values.min(), values.max(), 5).astype(int)

        # Calculate success rate per bin
        success_rate = [
            data[success_col][(values >= bins[i]) & (values < bins[i + 1])].mean() 
            for i in range(len(bins) - 1)
        ]

        # Plot bar chart
        plt.subplot(len(integer_columns), 1, idx)
        plt.bar(range(len(success_rate)), success_rate, tick_label=[f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)])
        plt.title(f"Success Rate by {column_name} (Integer)")
        plt.xlabel(f"{column_name} Ranges")
        plt.ylabel("Success Rate")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot float columns (line graphs)
    plt.figure(figsize=(15, len(float_columns) * 6))
    for idx, column in enumerate(float_columns, 1):
        values = data[column].dropna()
        column_name = column.replace('r_', '').replace('b_', '')  # Remove 'r_' and 'b_' prefixes
        bins = np.linspace(values.min(), values.max(), 7)

        # Calculate success rate per bin, ignoring empty bins
        success_rate = []
        bin_labels = []
        for i in range(len(bins) - 1):
            bin_data = data[success_col][(values >= bins[i]) & (values < bins[i + 1])]
            if not bin_data.empty:
                success_rate.append(bin_data.mean())
                bin_labels.append(f"{bins[i]:.2f}-{bins[i+1]:.2f}")

        # Plot line graph
        plt.subplot(len(float_columns), 1, idx)
        plt.plot(range(len(success_rate)), success_rate, marker='o')
        plt.title(f"Success Rate by {column_name} (Float)")
        plt.xlabel(f"{column_name} Ranges")
        plt.ylabel("Success Rate")
        plt.xticks(range(len(success_rate)), bin_labels, rotation=45)
    plt.tight_layout()
    plt.show()

# Function to visualize `previous_fights`
def visualize_previous_fights(data, success_col, column):
    plt.figure(figsize=(10, 6))
    values = data[column].dropna()
    bins = np.linspace(values.min(), values.max(), 5).astype(int)  # 4 bins (integer-based)

    # Calculate success rate per bin
    success_rate = []
    bin_labels = []
    for i in range(len(bins) - 1):
        bin_data = data[success_col][(values >= bins[i]) & (values < bins[i + 1])]
        if not bin_data.empty:
            success_rate.append(bin_data.mean())
            bin_labels.append(f"{bins[i]}-{bins[i+1]}")

    # Plot bar chart
    plt.bar(range(len(success_rate)), success_rate, tick_label=bin_labels)
    plt.title(f"Success Rate by {column}")
    plt.xlabel(f"{column} Ranges")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Select columns to analyze
columns_to_analyze = [
    col for col in data.columns if col.startswith('r_') or col.startswith('b_')
]

# Visualize original attributes
visualize_success_rate(data, 'success', columns_to_analyze)

# Visualize `previous_fights`
visualize_previous_fights(data_combined, 'success', 'previous_fights')

