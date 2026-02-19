import torch as T
import numpy as np
import pandas as pd
from environment import Env
from nnAgent import DQNAgent
import dataloader
import matplotlib.pyplot as plt

# Hyperparametreler
EPISODES = 500  # Eğitim epizodu sayısı
MAX_STEPS = 250  # Her epizodda maksimum adım sayısı
LEARNING_RATE = 0.005
DISCOUNT_FACTOR = 0.99
BUFFER_SIZE = 35000
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 250  # Kaç adımda bir target network güncellenecek

# Veriyi yükle ve çevreyi oluştur
data = dataloader.data
env = Env(data)

# Aracıyı oluştur
state_dim = 7  # d1, d2 ve d3 pozisyonları
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

        d1p, d2p, d3p, d4p, d5p, d6p, d7p = env.d1p, env.d2p, env.d3p, env.d4p, env.d5p, env.d6p, env.d7p
        avt_value = env._get_location(d1p, d2p, d3p, d4p, d5p, d6p, d7p)
        #avt = send_to_matlab(d1p, d2p, d3p)

        training_log.append({"Episode": episode + 1, "Step": step + 1, "d1p": d1p, "d2p": d2p, "d3p": d3p,"d4p": d4p,"d5p": d5p,"d6p": d6p,"d7p": d7p, "AVT": avt_value})

        agent.buffer.store(state, action, reward, next_state, done, env.avt_value, env.highest_avt, env.highest_cri)
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

