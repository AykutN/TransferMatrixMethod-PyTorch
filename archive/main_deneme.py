import torch as T
import numpy as np
import pandas as pd
from environment_deneme import Env
from nnAgent_deneme import DQNAgent
import matplotlib.pyplot as plt
import time
import signal  # Eğitim durdurma için sinyal işleme


# Hyperparametreler
EPISODES = 3000
MAX_STEPS = 250 
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
BUFFER_SIZE = 20000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 50 

# Eğitim durdurma için global değişken
stop_training = False

# Sinyal işleyici fonksiyon (Ctrl+C ile durdurma)
def signal_handler(sig, frame):
    global stop_training
    print("\nEğitim durduruluyor...")
    stop_training = True


signal.signal(signal.SIGINT, signal_handler)

env = Env()
state_dim = 7  
n_actions = env.n_actions
agent = DQNAgent(state_dim=state_dim, action_dim=n_actions, lr=LEARNING_RATE, gamma=DISCOUNT_FACTOR, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)



episode_rewards = []  
episode_losses = []  
average_positions = []  
final_positions = []  
training_log = []
pareto_solutions = []  # Pareto optimal çözümleri saklamak için liste

for episode in range(EPISODES):
    if stop_training:
        break  # Eğitim durdurulursa döngüden çık

    start_time = time.time()
    state = env.reset() 
    total_reward = 0
    done = False
    episode_loss = 0  
    positions = []  

    for step in range(MAX_STEPS):
        if stop_training:
            break  # Adım döngüsünü de durdur

        action = agent.select_action(state)  
        next_state, reward_vector, done = env.step(action)  

        d1p, d2p, d3p, d4p, d5p, d6p, d7p = env.d1p, env.d2p, env.d3p, env.d4p, env.d5p, env.d6p, env.d7p
        
        agent.update_avt_value(env.avt_value)

        # Eğitim logu
        training_log.append({
            "Episode": episode + 1, 
            "Step": step + 1, 
            "d1p": d1p, "d2p": d2p, "d3p": d3p,
            "d4p": d4p, "d5p": d5p, "d6p": d6p, "d7p": d7p, 
            "AVT": env.avt_value,
            "JPH": env.jph_value,
            "AVT Reward": reward_vector[0],
            "JPH Reward": reward_vector[1],
            "Total Reward": reward_vector[0] + reward_vector[1], 
            "Action": env.action_space[action]
        })

        # Pareto optimal çözümleri kaydetme
        if env.avt_value >= 25 and reward_vector[1] > 0:
            pareto_solutions.append({
                "Episode": episode + 1,
                "Step": step + 1,
                "State": state.tolist(),
                "Action": env.action_space[action],
                "AVT": env.avt_value,
                "JPH": env.jph_value,
                "Reward_AVT": reward_vector[0],
                "Reward_JPH": reward_vector[1]
            })

        agent.buffer.store(state, action, reward_vector, next_state, done)
        loss = agent.train()

        if loss is not None:
            episode_loss += loss 

        if agent.step % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        state = next_state
        total_reward += reward_vector[0] + reward_vector[1]
        positions.append(state) 

        if done:
            break

    average_position = np.mean(positions, axis=0)
    average_positions.append(average_position)
    final_positions.append(state)
    episode_rewards.append(total_reward)
    episode_losses.append(episode_loss)

    end_time = time.time()
    episode_duration = end_time - start_time

    print(f"Episode {episode + 1}/{EPISODES} - AVT Reward: {reward_vector[0]:.2f} - JPH Reward: {reward_vector[1]:.2f} - Total Reward: {total_reward:.2f} - Loss: {episode_loss:.4f} - Final Position: {state} - Final AVT: {env.avt_value:.4f} - Final JPH: {env.jph_value:.4f} - Duration: {episode_duration:.2f} seconds - Pareto Solutions Found: {len(pareto_solutions)}")

# Modeli kaydet
T.save(agent.q_network.state_dict(), "dqn_agent_model.pth")
print("Model başarıyla kaydedildi!")

# Eğitim loglarını kaydet
log_df = pd.DataFrame(training_log)
log_df.to_csv("training_log.csv", index=False)
print("Eğitim logları 'training_log.csv' dosyasına kaydedildi!")

# Pareto çözümleri kaydet
pareto_df = pd.DataFrame(pareto_solutions)
pareto_df.to_csv("pareto_solutions.csv", index=False)
print(f"Pareto optimal çözümler 'pareto_solutions.csv' dosyasına kaydedildi! Toplam {len(pareto_solutions)} çözüm bulundu.")

# Kayıp ve ödül loglarını kaydet
loss_df = pd.DataFrame({"Episode": range(1, len(episode_losses) + 1), "Loss": episode_losses})
loss_df.to_csv("loss_log.csv", index=False)
print("Kayıp değerleri 'loss_log.csv' dosyasına kaydedildi!")

reward_df = pd.DataFrame({"Episode": range(1, len(episode_rewards) + 1), "Total Reward": episode_rewards})
reward_df.to_csv("reward_log.csv", index=False)
print("Toplam ödül değerleri 'reward_log.csv' dosyasına kaydedildi!")