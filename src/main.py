import sys
import os
import torch as T
import numpy as np
import pandas as pd
import time
import csv

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import config.settings as config
from src.environment import Env
from src.agent import DQNAgent
import src.dataloader as dataloader

def main():
    # Make sure output dirs exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Initialize Loggers
    training_log_path = os.path.join(config.LOG_DIR, "training_log.csv")
    loss_log_path = os.path.join(config.LOG_DIR, "loss_log.csv")
    reward_log_path = os.path.join(config.LOG_DIR, "reward_log.csv")
    
    # Create CSV headers if files don't exist (overwrite mode for fresh start)
    with open(training_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Step", "d1p", "d2p", "d3p", "d4p", "d5p", "d6p", "AVT", "JPH", "Reward", "Action"])
        
    with open(loss_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Loss"])
        
    with open(reward_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total Reward"])

    # Initialize Env and Agent
    print("Initializing Environment...")
    env = Env()
    
    print("Initializing Agent...")
    state_dim = config.STATE_DIM
    n_actions = env.n_actions
    agent = DQNAgent(state_dim=state_dim, action_dim=n_actions)
    
    # data = dataloader.data # Referenced in original, kept if needed later
    
    print(f"Starting Training for {config.EPISODES} episodes...")
    
    try:
        for episode in range(config.EPISODES):
            state = env.reset()
            total_reward = 0
            done = False
            episode_loss = 0
            loss_count = 0
            
            # Buffer for step logs to write in batch per episode
            episode_step_logs = []
            
            for step in range(config.MAX_STEPS):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                d1p, d2p, d3p, d4p, d5p, d6p = env.d1p, env.d2p, env.d3p, env.d4p, env.d5p, env.d6p
                
                # Prepare log entry
                log_entry = [
                    episode + 1, step + 1,
                    d1p, d2p, d3p, d4p, d5p, d6p,
                    env.avt_value, env.jph_value,
                    reward,
                    env.action_space[action]
                ]
                episode_step_logs.append(log_entry)

                agent.buffer.store(state, action, reward, next_state, done, 
                                   env.avt_value, env.highest_avt, env.jph_value, env.highest_jph)
                
                loss = agent.train()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
                
                if agent.step % config.TARGET_UPDATE_FREQ == 0:
                    agent.update_target_network()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # End of Episode: Write logs
            with open(training_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(episode_step_logs)
                
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0
            with open(loss_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode + 1, avg_loss])
                
            with open(reward_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode + 1, total_reward])
                
            print(f"Ep {episode+1}/{config.EPISODES} | Reward: {total_reward:>7.2f} | AVT: {env.avt_value:>5.2f} | JPH: {env.jph_value:>5.2f} | D: [{env.d1p:5.1f}, {env.d2p:5.1f}, {env.d3p:5.1f}, {env.d4p:5.1f}, {env.d5p:5.1f}, {env.d6p:5.1f}]")
            
            # Save model occasionally
            if (episode + 1) % 50 == 0:
                T.save(agent.q_network.state_dict(), os.path.join(config.MODEL_DIR, "dqn_agent_model.pth"))

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final Save
        try:
            T.save(agent.q_network.state_dict(), os.path.join(config.MODEL_DIR, "dqn_agent_model_final.pth"))
            print("Model saved.")
        except:
            pass
        
        env.cleanup()
        print("Environment cleaned up.")

if __name__ == "__main__":
    main()
