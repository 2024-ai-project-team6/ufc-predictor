import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Load the datasets
fighter_stats_path = './fighter_stats.csv'
event_dataset_path = './event_dataset.csv'

fighter_stats = pd.read_csv(fighter_stats_path)
event_dataset = pd.read_csv(event_dataset_path)

event_dataset['date'] = pd.to_datetime(event_dataset['date'])
event_dataset = event_dataset[event_dataset['date'] >= '2022-01-01']

# Step 2: Split the dataset by weight class and sort by date
weight_classes = event_dataset['weight_class'].unique()
sorted_weight_class_datasets = {}

for weight_class in weight_classes:
    wc_data = event_dataset[event_dataset['weight_class'] == weight_class]
    wc_data = wc_data.sort_values(by='date')  # Sort by date
    sorted_weight_class_datasets[weight_class] = wc_data

# Step 3: Function to create a dynamic graph for each weight class
def create_dynamic_graph(wc_data, fighter_stats):
    dynamic_graphs = []
    G = nx.DiGraph()  # Directed graph for each weight class

    for index, row in wc_data.iterrows():
        r_fighter = row['r_fighter']
        b_fighter = row['b_fighter']

        # Add node for r_fighter if not exists
        if r_fighter not in G:
            r_data = fighter_stats[fighter_stats['name'] == r_fighter]
            if not r_data.empty:
                attributes = r_data.iloc[0].to_dict()
                del attributes['name']  # Remove identifier column
                attributes['win'] = 0
                attributes['lose'] = 0
                G.add_node(r_fighter, **attributes)

        # Add node for b_fighter if not exists
        if b_fighter not in G:
            b_data = fighter_stats[fighter_stats['name'] == b_fighter]
            if not b_data.empty:
                attributes = b_data.iloc[0].to_dict()
                del attributes['name']  # Remove identifier column
                attributes['win'] = 0
                attributes['lose'] = 0
                G.add_node(b_fighter, **attributes)

        # Update attributes: r_fighter wins, b_fighter loses
        if r_fighter in G:
            G.nodes[r_fighter]['win'] += 1
        if b_fighter in G:
            G.nodes[b_fighter]['lose'] += 1

        # Add edge from b_fighter to r_fighter
        G.add_edge(b_fighter, r_fighter, date=row['date'])

        # Save a copy of the graph for this time step
        dynamic_graphs.append(nx.DiGraph(G))

    return dynamic_graphs

# Step 4: Create dynamic graphs for each weight class
dynamic_graphs_by_weight_class = {}

for weight_class, wc_data in sorted_weight_class_datasets.items():
    dynamic_graphs_by_weight_class[weight_class] = create_dynamic_graph(wc_data, fighter_stats)

# Dynamic graphs are now prepared for each weight class.
# Further steps for LSTM-based graph network will follow.

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Step 5: Prepare Dataset for LSTM
class DynamicGraphDataset(Dataset):
    def __init__(self, dynamic_graphs):
        self.graph_sequences = self.process_graphs(dynamic_graphs)

    def process_graphs(self, dynamic_graphs):
        sequences = []
        for graph in dynamic_graphs:
            # Convert graph to adjacency matrix and node attributes
            adj_matrix = nx.adjacency_matrix(graph).todense()
            node_features = np.array([graph.nodes[node] for node in graph.nodes])
            sequences.append((adj_matrix, node_features))
        return sequences

    def __len__(self):
        return len(self.graph_sequences)

    def __getitem__(self, idx):
        return self.graph_sequences[idx]

# Prepare datasets for all weight classes
datasets_by_weight_class = {}
for weight_class, dynamic_graphs in dynamic_graphs_by_weight_class.items():
    datasets_by_weight_class[weight_class] = DynamicGraphDataset(dynamic_graphs)

def visualize_latest_dynamic_graphs(dynamic_graphs_by_weight_class, weight_class_to_visualize, num_graphs_to_show=10):
    """
    Visualizes the latest dynamic graphs for a specific weight class.

    Args:
    - dynamic_graphs_by_weight_class (dict): Dynamic graphs by weight class.
    - weight_class_to_visualize (str): The weight class to visualize.
    - num_graphs_to_show (int): Number of latest graphs to show for the selected weight class.
    """
    if weight_class_to_visualize not in dynamic_graphs_by_weight_class:
        print(f"Weight class {weight_class_to_visualize} not found!")
        return

    dynamic_graphs = dynamic_graphs_by_weight_class[weight_class_to_visualize]

    # Get the latest graphs
    num_graphs_to_show = min(num_graphs_to_show, len(dynamic_graphs))
    latest_graphs = dynamic_graphs[-num_graphs_to_show:]  # Get the last 'num_graphs_to_show' graphs

    plt.figure(figsize=(15, 5 * num_graphs_to_show))
    for i, graph in enumerate(latest_graphs):
        # Draw the graph
        plt.subplot(num_graphs_to_show, 1, i + 1)
        nx.draw(
            graph,
            with_labels=True,
            node_color="skyblue",
            edge_color="gray",
            node_size=1500,
            font_size=10
        )
        plt.title(f"Dynamic Graph for {weight_class_to_visualize} - Step {len(dynamic_graphs) - num_graphs_to_show + i + 1}")

    plt.tight_layout()
    plt.show()

# Example Usage
weight_class_to_visualize = "Lightweight"  # Change this to the desired weight class
visualize_latest_dynamic_graphs(dynamic_graphs_by_weight_class, weight_class_to_visualize, num_graphs_to_show=10)