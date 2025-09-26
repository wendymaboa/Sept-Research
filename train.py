"""Learning Tasks - Enhanced Training Script with Plotting"""
import numpy as np
import torch
import os
import sys
import itertools
import argparse
import gym
import gym_minigrid
import envs.envs
from envs.wrappers import *
from ppo_agent import PPOAgent
from library import *
import matplotlib.pyplot as plt
import json
from datetime import datetime

def plot_training_metrics(episode_returns, episode_successes, episode_steps, losses, save_path):
    """
    Plot training metrics and save them
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics', fontsize=16)
    
    episodes = range(len(episode_returns))
    
    # Plot 1: Reward vs Episodes
    axes[0, 0].plot(episodes, episode_returns, 'b-', alpha=0.3, label='Episode Reward')
    # Add moving average
    window_size = min(50, len(episode_returns))
    if len(episode_returns) >= window_size:
        moving_avg = np.convolve(episode_returns, np.ones(window_size)/window_size, mode='valid')
        axes[0, 0].plot(range(window_size-1, len(episode_returns)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Reward vs Episodes')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Steps vs Episodes
    axes[0, 1].plot(episodes, episode_steps, 'g-', alpha=0.7)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Steps vs Episodes')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Success Rate vs Episodes
    if len(episode_successes) >= window_size:
        success_moving_avg = np.convolve(episode_successes, np.ones(window_size)/window_size, mode='valid')
        axes[1, 0].plot(range(window_size-1, len(episode_successes)), success_moving_avg, 'm-', linewidth=2)
    axes[1, 0].scatter(episodes, episode_successes, alpha=0.5, s=10)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title(f'Success Rate vs Episodes (Moving Avg {window_size})')
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss vs Training Steps
    if losses:
        axes[1, 1].plot(losses, 'orange', linewidth=1)
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Loss vs Training Steps')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Loss Data Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Loss vs Training Steps')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(save_path, 'training_metrics.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to: {plot_filename}")
    
    # Also save raw data as JSON for later analysis
    data = {
        'episode_returns': episode_returns,
        'episode_successes': episode_successes,
        'episode_steps': episode_steps,
        'losses': losses,
        'timestamp': datetime.now().isoformat()
    }
    
    data_filename = os.path.join(save_path, 'training_data.json')
    with open(data_filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Training data saved to: {data_filename}")
    
    plt.show()

def evaluate(args):
    env, model, agent, num_episodes, max_episode_timesteps = args

    returns, successes = (0, 0)
    for _ in range(num_episodes):
        obs = env.reset()
        mission = tokenize(obs['mission'], model['vocab'])
        for t in range(max_episode_timesteps):
            # Different action selection based on agent type
            if hasattr(agent, 'actor_critic'):  # PPO agent
                action, _, _ = agent.get_action(obs['image'], mission)
            else:  # DQN agent
                action = select_action(model, obs['image'], mission)
            
            new_obs, reward, done, info = env.step(action)
            obs = new_obs
            returns += reward
            successes += (reward > 0) + 0
            if done:
                break
    return [returns, successes]

def train(env,
          agent_type='ppo',
          path='models',
          load_model=False,
          save_model=True,
          max_episodes=int(1e6),
          learning_starts=int(1e4),
          replay_buffer_size=int(1e5),
          train_freq=4,
          target_update_freq=int(1e3),
          batch_size=32,
          gamma=0.95,
          learning_rate=1e-4,
          eps_initial=1.0,
          eps_final=0.1,
          eps_success=0.98,
          timesteps_success=50,
          mean_episodes=50,
          eps_timesteps=int(5e5),
          print_freq=5,
          # PPO specific parameters
          ppo_epochs=4,
          clip_epsilon=0.2,
          gae_lambda=0.95):
    
    ### Initialising
    eps_schedule = LinearSchedule(eps_timesteps, eps_final, eps_initial)
    
    if agent_type == 'dqn':
        replay_buffer = ReplayBuffer(replay_buffer_size, batch_size)
        agent = Agent(env, gamma=gamma, learning_rate=learning_rate, 
                     replay_buffer=replay_buffer, path=path)
    elif agent_type == 'ppo':
        agent = PPOAgent(env, learning_rate=learning_rate, gamma=gamma,
                        gae_lambda=gae_lambda, clip_epsilon=clip_epsilon,
                        ppo_epochs=ppo_epochs, batch_size=batch_size)
    
    # Create consistent vocabulary
    vocab = Vocabulary(100)
    agent.vocab = vocab
    
    # Load model if requested
    if load_model and os.path.exists(path):
        try:
            if agent_type == 'dqn':
                model_data = load(path, env)
                agent.vocab = model_data['vocab']
                agent.q_func.load_state_dict(model_data['params'].state_dict())
                agent.target_q_func.load_state_dict(model_data['params'].state_dict())
            else:  # PPO
                checkpoint = torch.load(path)
                agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
                agent.vocab.load_vocab(checkpoint['vocab'])
            print(f'{agent_type.upper()} model loaded from {path}')
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Starting training from scratch")
    
    # Set up model reference
    if agent_type == 'dqn':
        model = {'params': agent.q_func, 'vocab': agent.vocab}
    else:
        model = {'params': agent.actor_critic, 'vocab': agent.vocab}
    
    agent.path = path
    
    # Training loop tracking variables
    episode_returns = []
    episode_successes = []
    episode_steps = []  # Track steps per episode
    losses = []  # Track training losses
    avg_return = 0
    success_rate = 0
    avg_return_best = -float('inf')
    success_rate_best = 0.0
    steps = 0
    
    # Create directory for saving plots and data
    save_dir = os.path.dirname(path)
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        for episode in range(max_episodes):
            obs = env.reset()
            mission = tokenize(obs['mission'], agent.vocab)
            
            episode_returns.append(0.0)
            episode_successes.append(0.0)
            episode_step_count = 0
            done = False
            t = 0
            
            while not done and t < timesteps_success:
                # Action selection
                if agent_type == 'dqn':
                    if random.random() > eps_schedule(steps):
                        action = select_action(model, obs['image'], mission)
                    else:
                        action = env.action_space.sample()
                    
                    # Store in replay buffer
                    new_obs, reward, done, info = env.step(action)
                    replay_buffer.add(mission, obs['image'], action, reward, new_obs['image'], done, info)
                    
                    # Update DQN and track loss
                    if steps > learning_starts and steps % train_freq == 0:
                        loss = agent.update_td_loss()
                        if loss is not None:
                            losses.append(loss)
                    
                    if steps > learning_starts and steps % target_update_freq == 0:
                        agent.update_target_network()
                    
                else:  # PPO
                    try:
                        action, log_prob, value = agent.get_action(obs['image'], mission)
                        new_obs, reward, done, info = env.step(action)
                        
                        # Store transition for PPO
                        agent.store_transition(obs['image'], mission, action, log_prob, value, reward, done)
                    except Exception as e:
                        print(f"Error in PPO action/storage: {e}")
                        # Take random action as fallback
                        action = env.action_space.sample()
                        new_obs, reward, done, info = env.step(action)
                
                # Common updates
                obs = new_obs
                episode_returns[-1] += reward
                episode_successes[-1] = (reward > 0)
                episode_step_count += 1
                
                t += 1
                steps += 1
            
            # Store episode step count
            episode_steps.append(episode_step_count)
            
            # PPO update at end of episode
            if agent_type == 'ppo' and len(agent.episode_data['rewards']) > 0:
                try:
                    ppo_loss = agent.update()
                    if ppo_loss is not None:
                        losses.append(ppo_loss)
                except Exception as e:
                    print(f"Error in PPO update: {e}")
                    agent.clear_memory()  # Clear memory on error
            
            # Evaluation and saving
            if episode % 100 == 0 and episode > 0:
                print("evaluating ...")
                try:
                    eval_env = env
                    args = [eval_env, model, agent, mean_episodes, timesteps_success]
                    returns, successes = evaluate(args)
                    avg_return, success_rate = (returns/mean_episodes, successes/mean_episodes)
                    
                    if success_rate > success_rate_best:
                        avg_return_best = avg_return
                        success_rate_best = success_rate
                        if save_model:
                            try:
                                agent.save()
                                print(f"\nModels saved to {path}. ar: {avg_return_best}, sr: {success_rate_best}\n")
                            except Exception as e:
                                print(f"Error saving model: {e}")
                    
                    if success_rate_best > eps_success:
                        print(f"\nTask solved: success_rate > {eps_success}\n")  
                        break
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    
            ### Print training progress
            if episode % print_freq == 0:
                window = min(mean_episodes, len(episode_returns))
                avg_return_ = round(np.mean(episode_returns[-window:]), 1)
                success_rate_ = np.mean(episode_successes[-window:])
                avg_steps = np.mean(episode_steps[-window:]) if episode_steps else 0
                
                print("--------------------------------------------------------")
                print(f"steps {steps}")
                print(f"episodes {episode}")
                print(f"mission {obs['mission']}")
                print(f"average return: current {avg_return_}, eval_current {avg_return}, eval_best {avg_return_best}")
                print(f"success rate: current {success_rate_}, eval_current {success_rate}, eval_best {success_rate_best}")
                print(f"average episode steps: {avg_steps:.1f}")
                if agent_type == 'dqn':
                    print(f"% time spent exploring {int(100 * eps_schedule(steps))}")
                if losses:
                    print(f"recent loss: {losses[-1]:.4f}")
                print("--------------------------------------------------------")
        
        # Plot training metrics at the end
        print("Generating training plots...")
        plot_training_metrics(episode_returns, episode_successes, episode_steps, losses, save_dir)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if save_model:
            try:
                agent.save()
                print("Model saved before exit")
            except Exception as e:
                print(f"Error saving model on interrupt: {e}")
        
        # Still generate plots with available data
        if episode_returns:
            print("Generating training plots with available data...")
            plot_training_metrics(episode_returns, episode_successes, episode_steps, save_dir)
    
    except Exception as e:
        print(f"Training error: {e}")
        raise
    
    return agent, model, episode_returns, episode_successes

parser = argparse.ArgumentParser()
parser.add_argument(
    '--env_key',
    default="Minigrid-PickUpObj-Custom-v0",
    help="Environment"
)
parser.add_argument(
    '--egocentric',
    default=False,
    help="Egocentric or allocentric",
    action='store_true'
)
parser.add_argument(
    '--agent_type',
    default="dqn",
    choices=["dqn", "ppo"],
    help="Type of agent to use"
)
parser.add_argument(
    '--view_size',
    type=int,
    default=None,
    help="View size for partial observation (None for full observability)"
)
parser.add_argument(
    '--partial_egocentric',
    action='store_true',
    default=False,
    help="Use egocentric partial observation"
)
parser.add_argument(
    '--load_model',
    action='store_true',
    default=False,
    help="Load existing model"
)
parser.add_argument(
    '--save_model',
    action='store_true',
    default=False,
    help="Save model during training"
)

if __name__ == '__main__':    
    args = parser.parse_args()
    
    # Use the custom environment for PPO testing
    if args.env_key == "MiniGrid-Empty-5x5-v0":
        print("Using MiniGrid-Empty-5x5-v0 environment")
        env = gym.make(args.env_key)
    else:
        env = gym.make(args.env_key)
    
    # Apply observation wrappers based on arguments
    if args.view_size is not None:
        # Partial observation
        env = PartialObsWrapper(env, view_size=args.view_size, 
                               egocentric=args.partial_egocentric)
    else:
        # Full observation
        env = FullyObsWrapper(env, egocentric=args.egocentric)
    
    env = RGBImgObsWrapper(env)
    
    # Determine model save path based on observation type
    base_folder = 'modelsego' if args.egocentric or args.partial_egocentric else 'models'
    
    # Create more descriptive filename
    obs_type = ""
    if args.view_size is not None:
        obs_type = f"_partial{args.view_size}"
        if args.partial_egocentric:
            obs_type += "_ego"
    elif args.egocentric:
        obs_type = "_ego"
    else:
        obs_type = "_allo"
    
    path = f'{base_folder}/{args.env_key}_{args.agent_type}{obs_type}'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    print(f"Training {args.agent_type.upper()} agent on {args.env_key}")
    print(f"Observation type: {'Egocentric' if (args.egocentric or args.partial_egocentric) else 'Allocentric'}")
    if args.view_size is not None:
        print(f"View size: {args.view_size}")
    print(f"Observation shape: {env.observation_space.spaces['image'].shape}")
    print(f"Action space: {env.action_space.n}")
    print(f"Model will be saved to: {path}")
    
    train(env, 
          agent_type=args.agent_type,
          path=path, 
          save_model=args.save_model, 
          load_model=args.load_model)