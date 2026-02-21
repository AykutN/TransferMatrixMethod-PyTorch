import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
log_dir = "outputs/logs"
loss_path = os.path.join(log_dir, "loss_log.csv")
reward_path = os.path.join(log_dir, "reward_log.csv")
output_plot = "outputs/logs/training_metrics.png"

# Read Data
df_loss = pd.read_csv(loss_path)
df_reward = pd.read_csv(reward_path)

# Plotting
plt.figure(figsize=(12, 10))

# 1. Loss Plot
plt.subplot(2, 1, 1)
plt.plot(df_loss['Episode'], df_loss['Loss'], color='#E63946', label='Loss', linewidth=1.5)
plt.title('Training Loss per Episode', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 2. Reward Plot
plt.subplot(2, 1, 2)
plt.plot(df_reward['Episode'], df_reward['Total Reward'], color='#457B9D', label='Total Reward', linewidth=1.5)
plt.title('Total Reward per Episode', fontsize=14, fontweight='bold')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(output_plot, dpi=300)
print(f"Plot saved to {output_plot}")
plt.show()
