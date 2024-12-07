import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = './체급별_결과.csv'
data = pd.read_csv(file_path)

# Ensure numeric columns are properly converted
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='ignore')

# Combine `r_` and `b_` columns into unified columns
columns_to_combine = [col[2:] for col in data.columns if col.startswith('r_')]
combined_data = pd.DataFrame()

for col in columns_to_combine:
    r_col = f"r_{col}"
    b_col = f"b_{col}"
    combined_data[col] = pd.concat([data[r_col], data[b_col]], ignore_index=True)

# Add success column (duplicated for red and blue)
combined_data['success'] = pd.concat([data['success'], data['success']], ignore_index=True)

# Function to visualize success rate by multiple attributes
def visualize_success_rate(data, success_col):
    integer_columns = []
    float_columns = []

    # Separate columns into integer-like and float types
    for column in data.columns:
        if column == success_col:
            continue
        if np.issubdtype(data[column].dtype, np.integer):
            integer_columns.append(column)
        elif np.issubdtype(data[column].dtype, np.float64) or np.issubdtype(data[column].dtype, np.float32):
            # Treat float columns as integer if all values are essentially integers
            if (data[column].dropna() % 1 == 0).all():
                integer_columns.append(column)
            else:
                float_columns.append(column)

    # Plot integer columns (bar charts)
    plt.figure(figsize=(15, len(integer_columns) * 6))
    for idx, column in enumerate(integer_columns, 1):
        values = data[column].dropna()
        column_name = column  # Column name without modification
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
        column_name = column  # Column name without modification
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

# Visualize success rates for combined data
visualize_success_rate(combined_data, 'success')
