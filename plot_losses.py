import matplotlib.pyplot as plt
import argparse
import os
import re

def parse_loss_file(log_path):
    """
    Parse the loss log file and extract epoch, train loss, val loss, and learning rate.
    
    Args:
        log_path (str): Path to the log file
        
    Returns:
        dict: Dictionary containing lists of epochs, train_losses, val_losses, and learning_rates
    """
    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse line using regex
            match = re.search(r'Epoch \[(\d+)/\d+\], Loss Train\.: ([\d.]+), Loss Val\.: ([\d.]+), Learning rate: ([\d.e+-]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))
                learning_rates.append(float(match.group(4)))
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates
    }

def plot_losses(model_name):
    """
    Plot training and validation losses for a given model.
    
    Args:
        model_name (str): Name of the model (e.g., DSCMS, PRUSR)
    """
    log_path = f'./models/{model_name}/output/logs/{model_name}_loss.txt'
    
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    # Parse the log file
    data = parse_loss_file(log_path)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot losses
    ax1.plot(data['epochs'], data['train_losses'], label='Train Loss', linewidth=2)
    ax1.plot(data['epochs'], data['val_losses'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax2.plot(data['epochs'], data['learning_rates'], color='green', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title(f'{model_name} - Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')  # Log scale for learning rate
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = f'./models/{model_name}/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Loss plot saved to {output_path}")
    print(f"Total epochs: {len(data['epochs'])}")
    print(f"Final train loss: {data['train_losses'][-1]:.4f}")
    print(f"Final val loss: {data['val_losses'][-1]:.4f}")
    print(f"Best val loss: {min(data['val_losses']):.4f} at epoch {data['epochs'][data['val_losses'].index(min(data['val_losses']))]}")

def plot_all_models(model_list):
    """
    Plot training and validation losses for all models in a single comparison plot.
    
    Args:
        model_list (list): List of model names
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = plt.cm.tab10(range(len(model_list)))
    
    for idx, model_name in enumerate(model_list):
        log_path = f'./models/{model_name}/output/logs/{model_name}_loss.txt'
        
        if not os.path.exists(log_path):
            print(f"Warning: Log file not found for {model_name}: {log_path}")
            continue
        
        # Parse the log file
        data = parse_loss_file(log_path)
        
        # Plot losses
        ax1.plot(data['epochs'], data['train_losses'], 
                label=f'{model_name.upper()} - Train', 
                linewidth=2, color=colors[idx], linestyle='-')
        ax1.plot(data['epochs'], data['val_losses'], 
                label=f'{model_name.upper()} - Val', 
                linewidth=2, color=colors[idx], linestyle='--')
        
        # Plot learning rate
        ax2.plot(data['epochs'], data['learning_rates'], 
                label=f'{model_name.upper()}', 
                linewidth=2, color=colors[idx])
        
        # Print stats
        print(f"\n{model_name.upper()} Statistics:")
        print(f"  Total epochs: {len(data['epochs'])}")
        print(f"  Final train loss: {data['train_losses'][-1]:.4f}")
        print(f"  Final val loss: {data['val_losses'][-1]:.4f}")
        print(f"  Best val loss: {min(data['val_losses']):.4f} at epoch {data['epochs'][data['val_losses'].index(min(data['val_losses']))]}")
    
    # Configure loss plot
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Comparison - Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Configure learning rate plot
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Model Comparison - Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = './models/comparison'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'all_models_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nComparison plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training losses for Super Resolution Model')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., DSCMS, PRUSR) or "all" for comparison plot')
    args = parser.parse_args()
    
    # Model list
    MODEL_LIST = ['dscms', 'prusr']
    
    if args.model.lower() == 'all':
        plot_all_models(MODEL_LIST)
    else:
        plot_losses(args.model.lower())