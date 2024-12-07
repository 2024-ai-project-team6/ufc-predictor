### 각 컬럼별로 success rate 시각화
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = './Lightweight_results.csv'
data = pd.read_csv(file_path)

# Ensure numeric columns are properly converted
for column in data.columns:
    try:
        data[column] = pd.to_numeric(data[column], errors='coerce')  # Convert to numeric or NaN
    except:
        pass

# Combine `r_` and `b_` columns into unified columns
columns_to_combine = [col[2:] for col in data.columns if col.startswith('r_')]
combined_data = pd.DataFrame()

# Combine `r_` and `b_` columns
for col in columns_to_combine:
    r_col = f"r_{col}"
    b_col = f"b_{col}"
    combined_data[col] = pd.concat([data[r_col], data[b_col]], ignore_index=True)

# Include `success` column
combined_data['success'] = pd.concat([data['success'], data['success']], ignore_index=True)

# Add `_diff` columns to the dataset without merging
diff_columns = [col for col in data.columns if col.endswith('_diff')]
for col in diff_columns:
    combined_data[col] = data[col]

# Special handling for 'age' column
if 'age' in combined_data.columns:
    combined_data['age'] = combined_data['age'].round().astype(int)  # Convert age to integer


# Function to visualize success rate with bins
def visualize_success_rate(data, success_col, max_bins=6):  # Max bins set to 6
    for column in data.columns:
        if column == success_col:
            continue

        # Drop NaN values
        column_data = data[[column, success_col]].dropna()

        # Determine bins based on column type
        if column == 'age' or np.issubdtype(column_data[column].dtype, np.integer):  # Integer type or age
            unique_values = sorted(column_data[column].unique())
            if len(unique_values) > max_bins:  # Compress bins to max_bins
                bins = np.linspace(min(unique_values), max(unique_values), max_bins + 1).astype(int)
            else:
                bins = unique_values
        else:  # Float or continuous type
            bins = np.linspace(column_data[column].min(), column_data[column].max(), max_bins + 1)

        # Calculate success rate for each bin
        success_rate = []
        bin_labels = []
        for i in range(len(bins) - 1):
            bin_data = column_data[(column_data[column] >= bins[i]) & (column_data[column] < bins[i + 1])]
            if not bin_data.empty:
                success_rate.append(bin_data[success_col].mean())
                bin_labels.append(f"{bins[i]:.2f}-{bins[i + 1]:.2f}" if not np.issubdtype(column_data[column].dtype, np.integer) else f"{int(bins[i])}-{int(bins[i + 1])}")

        # Plot success rate for each bin
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(success_rate)), success_rate, marker='o', linestyle='-')
        plt.xticks(range(len(success_rate)), bin_labels, rotation=45)
        plt.title(f"Success Rate by {column} (Binned)")
        plt.xlabel("Bins")
        plt.ylabel("Success Rate")
        plt.tight_layout()
        plt.show()


# Visualize original attributes (combined columns)
visualize_success_rate(combined_data, 'success', max_bins=6)
