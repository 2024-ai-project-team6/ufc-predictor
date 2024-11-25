import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_losses, weight_class):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.title(f'Training History - {weight_class}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'training_history_{weight_class}.png')
    plt.close()