import torch as T
import numpy as np
import pandas as pd
from environment import Env
from nnAgent import DQNAgent
import dataloader
import matplotlib.pyplot as plt
import socket
import struct
import time



# Hyperparametreler
EPISODES = 500  # Eğitim epizodu sayısı
MAX_STEPS = 250  # Her epizodda maksimum adım sayısı
LEARNING_RATE = 0.005
DISCOUNT_FACTOR = 0.99
BUFFER_SIZE = 50000
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 250  # Kaç adımda bir target network güncellenecek

# Veriyi yükle ve çevreyi oluştur
data = dataloader.data
env = Env(data)

# Aracıyı oluştur
state_dim = 3  # d1, d2 ve d3 pozisyonları
n_actions = env.n_actions
agent = DQNAgent(state_dim=state_dim, action_dim=n_actions, lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

episode_rewards = []  
episode_losses = []  
average_positions = []  
final_positions = []  
training_log = []

for episode in range(EPISODES):
    state = env.reset()  # Ortamı sıfırla
    total_reward = 0
    done = False
    episode_loss = 0  
    positions = []  

    for step in range(MAX_STEPS):
        action = agent.select_action(state)  # Aksiyon seç
        next_state, reward, done = env.step(action)  # Ortamı bir adım ilerlet

        d1p, d2p, d3p = env.d1p, env.d2p, env.d3p
        avt_value, cri_value = env._get_location(d1p, d2p, d3p)
        
        #avt_value, cri_value = send_to_matlab(d1p, d2p, d3p)

        training_log.append({"Episode": episode + 1, "Step": step + 1, "d1p": d1p, "d2p": d2p, "d3p": d3p, "AVT": avt_value, "CRI": cri_value})

        agent.buffer.store(state, action, reward, next_state, done, env.avt_value, env.cri_value, env.highest_avt, env.highest_cri)
        loss = agent.train()

        if loss is not None:
            episode_loss += loss 

        if agent.step % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        state = next_state  # Durumu güncelle
        total_reward += reward
        positions.append(state)  # Konumu sakla

        if done:
            break

    average_position = np.mean(positions, axis=0)
    average_positions.append(average_position)
    final_positions.append(state)
    episode_rewards.append(total_reward)
    episode_losses.append(episode_loss)

    print(f"Episode {episode + 1}/{EPISODES} - Total Reward: {total_reward:.2f} - Loss: {episode_loss:.4f} - Average Position: {average_position} - Final Position: {state} - Final AVT: {env._get_location(state[0], state[1], state[2])}")

# Modeli kaydet
T.save(agent.q_network.state_dict(), "dqn_agent_model.pth")
print("Model başarıyla kaydedildi!")

# Dataframe'e çevir ve CSV olarak kaydet
log_df = pd.DataFrame(training_log)
log_df.to_csv("training_log.csv", index=False)
print("Eğitim logları 'training_log.csv' dosyasına kaydedildi!")

# Eğitim sonuçlarını görselleştir
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Performance")
plt.legend()
plt.savefig("training_performance.png")  # Grafiği dosyaya kaydeder
plt.show()

# Loss değişimini görselleştir
plt.figure(figsize=(10, 5))
plt.plot(episode_losses, label="Total Loss per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Loss")
plt.title("DQN Training Loss")
plt.legend()
plt.savefig("training_loss.png")  # Grafiği dosyaya kaydeder
plt.show()

# Ortalama durulan konumu ve bitirilen konumu yazdır
average_final_position = np.mean(final_positions, axis=0)
print(f"Average Final Position: d1={average_final_position[0]}, d2={average_final_position[1]}, d3={average_final_position[2]}")

# Final konumların scatter plot'unu oluştur
final_positions_array = np.array(final_positions)

plt.figure(figsize=(10, 8))
plt.scatter(final_positions_array[:, 0], final_positions_array[:, 1], c=final_positions_array[:, 2], marker='o', alpha=0.6)
plt.colorbar(label='d3 Position')
plt.xlabel('d1 Position')
plt.ylabel('d2 Position')
plt.title('Scatter Plot of Final Positions')
plt.savefig("final_positions_scatter.png")  # Scatter plot'u dosyaya kaydeder
plt.show()

# Final AVT ve CRI'lerin histogramını oluştur
final_avt_positions = [env._get_location(pos[0], pos[1], pos[2]) for pos in final_positions if isinstance(pos, np.ndarray) and len(pos) == 3]
final_avt_positions_array = np.array(final_avt_positions)

plt.figure(figsize=(10, 8))
plt.hist(final_avt_positions_array[:, 0], bins=50, color='green', alpha=0.7, label='AVT')
plt.hist(final_avt_positions_array[:, 1], bins=50, color='blue', alpha=0.7, label='CRI')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Final AVT and CRI Values')
plt.legend()
plt.savefig("final_avt_cri_histogram.png")  # Histogramı dosyaya kaydeder
plt.show()

# Isı haritası oluştur
heatmap_data = np.zeros((100, 100))

for pos in final_positions:
    if isinstance(pos, np.ndarray) and len(pos) == 3:
        d1_idx = int(pos[0] * 10)  # d1 pozisyonunu 0-99 aralığına ölçekle
        d2_idx = int(pos[1] * 10)  # d2 pozisyonunu 0-99 aralığına ölçekle
        d3_idx = int(pos[2] * 10)  # d3 pozisyonunu 0-99 aralığına ölçekle
        if 0 <= d1_idx < 100 and 0 <= d2_idx < 100 and 0 <= d3_idx < 100:
            heatmap_data[d1_idx, d2_idx] += d3_idx  # d3 pozisyonunu frekans olarak ekle

plt.figure(figsize=(10, 8))
plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
plt.colorbar(label='d3 Position (scaled)')
plt.xlabel('d1 Position (scaled)')
plt.ylabel('d2 Position (scaled)')
plt.title('Heatmap of Final Positions')
plt.savefig("final_positions_heatmap.png")  # Isı haritasını dosyaya kaydeder
plt.show()

# Her 100 epizod için ortalama AVT ve CRI değerini hesapla ve sakla
average_avt_per_100_episodes = []
average_cri_per_100_episodes = []
max_avt_positions = []

for i in range(0, EPISODES, 100):
    avt_values = [env._get_location(pos[0], pos[1], pos[2]) for pos in final_positions[i:i+100] if isinstance(pos, np.ndarray) and len(pos) == 3]
    if avt_values:
        average_avt = np.mean([avt[0] for avt in avt_values])
        average_cri = np.mean([avt[1] for avt in avt_values])
        average_avt_per_100_episodes.append(average_avt)
        average_cri_per_100_episodes.append(average_cri)
        max_avt_index = np.argmax([avt[0] for avt in avt_values])
        max_avt_positions.append(final_positions[i + max_avt_index])

# Ortalama AVT ve CRI değerlerini çizgi grafiği ile göster
plt.figure(figsize=(10, 5))
plt.plot(range(0, EPISODES, 100), average_avt_per_100_episodes, marker='o', label="Average AVT per 100 Episodes")
plt.plot(range(0, EPISODES, 100), average_cri_per_100_episodes, marker='o', label="Average CRI per 100 Episodes")
plt.xlabel("Episode")
plt.ylabel("Average Value")
plt.title("Average AVT and CRI per 100 Episodes")
plt.legend()
plt.savefig("average_avt_cri_per_100_episodes.png")  # Grafiği dosyaya kaydeder
plt.show()

# Maksimum AVT değerinin olduğu d1p, d2p, d3p noktalarını yazdır
for i, pos in enumerate(max_avt_positions):
    print(f"Max AVT Position for Episodes {i*100+1}-{(i+1)*100}: d1={pos[0]}, d2={pos[1]}, d3={pos[2]}")