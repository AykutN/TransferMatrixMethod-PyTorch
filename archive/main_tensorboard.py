import torch as T
import numpy as np
import pandas as pd
from environment_6 import Env
from nnAgent_6 import DQNAgent
import dataloader
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime

# Hyperparametreler
EPISODES = 3000
MAX_STEPS = 250 
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
BUFFER_SIZE = 20000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100 

# TensorBoard writer oluştur
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/DQN_training_{timestamp}')

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
        episode_start = time.time()
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
        episode_time = time.time() - episode_start
        average_position = np.mean(positions, axis=0)
        average_positions.append(average_position)
        final_positions.append(state)
        episode_rewards.append(total_reward)
        episode_losses.append(episode_loss)

        # TensorBoard'a metrikleri yaz
        writer.add_scalar('Training/Episode Reward', total_reward, episode)
        writer.add_scalar('Training/Episode Loss', episode_loss, episode)
        writer.add_scalar('Training/AVT', env.avt_value, episode)
        writer.add_scalar('Training/JPH', env.jph_value, episode)
        writer.add_scalar('Training/Episode Time', episode_time, episode)
        writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
        
        # Her pozisyon için ayrı grafik
        for i, pos in enumerate(average_position):
            writer.add_scalar(f'Positions/Position_{i+1}', pos, episode)

        # AVT vs JPH scatter plot için
        writer.add_scalars('Performance Metrics', {
            'AVT': env.avt_value,
            'JPH': env.jph_value
        }, episode)

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
    
    # TensorBoard writer'ı kapat
    writer.close()