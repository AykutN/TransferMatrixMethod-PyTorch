import csv
import matplotlib.pyplot as plt
import os

# Paths
log_dir = "outputs/logs"
loss_path = os.path.join(log_dir, "loss_log.csv")
reward_path = os.path.join(log_dir, "reward_log.csv")
output_plot = "outputs/logs/training_metrics.png"

def read_csv(path, col_name):
    data = []
    eps = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps.append(int(row['Episode']))
            data.append(float(row[col_name]))
    return eps, data

# Read Data
try:
    eps_loss, loss_vals = read_csv(loss_path, 'Loss')
    eps_reward, reward_vals = read_csv(reward_path, 'Total Reward')

    # Plotting
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(eps_loss, loss_vals, color='red', label='Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(eps_reward, reward_vals, color='blue', label='Total Reward')
    plt.title('Total Reward')
    plt.xlabel('Episode')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
except Exception as e:
    print(f"Error: {e}")
