import torch as T
import numpy as np
import pandas as pd
from environment_6 import Env
from nnAgent_6 import DQNAgent
import dataloader
import matplotlib
matplotlib.use('TkAgg')  # macOS için backend ayarı
import matplotlib.pyplot as plt

# Hyperparametreler
EPISODES = 3000
MAX_STEPS = 250 
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
BUFFER_SIZE = 20000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100 

# Grafik ayarları
plt.ion()  
plt.style.use('dark_background')  
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Training Progress', fontsize=14)


episodes_list = []
rewards_list = []
losses_list = []
avt_list = []
jph_list = []


reward_line, = ax1.plot(episodes_list, rewards_list, 'b-', label='Reward', linewidth=2)
loss_line, = ax2.plot(episodes_list, losses_list, 'r-', label='Loss', linewidth=2)
scatter = ax3.scatter([], [], c='g', label='AVT vs JPH', s=50)

for ax in [ax1, ax2, ax3]:
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(labelsize=10)

ax1.set_xlabel('Episode', fontsize=10)
ax1.set_ylabel('Total Reward', fontsize=10)
ax1.set_title('Reward Progress', fontsize=12)
ax1.legend(fontsize=10)

ax2.set_xlabel('Episode', fontsize=10)
ax2.set_ylabel('Loss', fontsize=10)
ax2.set_title('Loss Progress', fontsize=12)
ax2.legend(fontsize=10)

ax3.set_xlabel('Final AVT', fontsize=10)
ax3.set_ylabel('Final JPH', fontsize=10)
ax3.set_title('AVT vs JPH', fontsize=12)
ax3.legend(fontsize=10)

plt.show(block=False)

def update_plots():
    reward_line.set_data(episodes_list, rewards_list)
    loss_line.set_data(episodes_list, losses_list)
    scatter.set_offsets(np.c_[avt_list, jph_list])
    
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()
    
    try:
        fig.canvas.flush_events()
        plt.pause(0.1)  
    except Exception as e:
        print(f"Plot güncelleme hatası: {e}")

data = dataloader.data
env = Env()

state_dim = 6  
n_actions = env.n_actions
agent = DQNAgent(state_dim=state_dim, action_dim=n_actions, lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

episode_rewards = []  
episode_losses = []  
average_positions = []  
final_positions = []  
training_log = []

try:
    for episode in range(EPISODES):
        state = env.reset() 
        total_reward = 0
        done = False
        episode_loss = 0  
        positions = []  

        for step in range(MAX_STEPS):
            action = agent.select_action(state)  
            next_state, reward, done = env.step(action)  

            d1p, d2p, d3p, d4p, d5p, d6p = env.d1p, env.d2p, env.d3p, env.d4p, env.d5p, env.d6p
            
            training_log.append({
                "Episode": episode + 1, 
                "Step": step + 1, 
                "d1p": d1p, "d2p": d2p, "d3p": d3p,
                "d4p": d4p, "d5p": d5p, "d6p": d6p,
                "AVT": env.avt_value,
                "JPH": env.jph_value,
                "Reward": reward, 
                "Action": env.action_space[action]
            })
            
            agent.buffer.store(state, action, reward, next_state, done, env.avt_value, env.highest_avt, env.jph_value, env.highest_jph)
            loss = agent.train()

            if loss is not None:
                episode_loss += loss 

            if agent.step % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            state = next_state
            total_reward += reward
            positions.append(state) 

            if done:
                break

        # Episode sonu işlemleri
        average_position = np.mean(positions, axis=0)
        average_positions.append(average_position)
        final_positions.append(state)
        episode_rewards.append(total_reward)
        episode_losses.append(episode_loss)
        
        # Grafik verilerini güncelle
        episodes_list.append(episode + 1)
        rewards_list.append(total_reward)
        losses_list.append(episode_loss)
        avt_list.append(env.avt_value)
        jph_list.append(env.jph_value)
        
        # Grafikleri güncelle
        update_plots()

        print(f"Episode {episode + 1}/{EPISODES} - Total Reward: {total_reward:.2f} - Loss: {episode_loss:.4f} - Average Position: {average_position} - Final Position: {state} - Final AVT: {env.avt_value} - Final JPH: {env.jph_value}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user! Saving current progress...")

finally:
    # Modeli kaydet
    T.save(agent.q_network.state_dict(), "dqn_agent_model.pth")
    print("Model başarıyla kaydedildi!")

    # Dataframe'e çevir ve CSV olarak kaydet
    log_df = pd.DataFrame(training_log)
    log_df.to_csv("training_log.csv", index=False)
    print("Eğitim logları 'training_log.csv' dosyasına kaydedildi!")

    # Kayıp değerlerini kaydet
    loss_df = pd.DataFrame({"Episode": range(1, len(episode_losses) + 1), "Loss": episode_losses})
    loss_df.to_csv("loss_log.csv", index=False)
    print("Kayıp değerleri 'loss_log.csv' dosyasına kaydedildi!")

    # Toplam ödül değerlerini kaydet
    reward_df = pd.DataFrame({"Episode": range(1, len(episode_rewards) + 1), "Total Reward": episode_rewards})
    reward_df.to_csv("reward_log.csv", index=False)
    print("Toplam ödül değerleri 'reward_log.csv' dosyasına kaydedildi!")
    
    # Grafikleri kaydet
    plt.savefig("training_plots.png")
    plt.close()