"""
Visualising Learned Tasks - Fixed for PPO compatibility
"""

import torch
import numpy as np
import os
import itertools
import argparse
import random
import gym
import gym_minigrid

# Import your custom modules
from library import *
from ppo_agent import PPOAgent
import envs.envs
from envs.wrappers import *
from matplotlib import pyplot as plt
from gym_minigrid.window import Window
import imageio

def load_ppo_model(path, env):
    """Load PPO model with proper structure"""
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    
    # Create PPO agent
    agent = PPOAgent(env)
    
    # Load model state
    agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    
    # Load vocabulary
    vocab = Vocabulary(100)
    vocab.load_vocab(checkpoint['vocab'])
    
    return {'params': agent.actor_critic, 'vocab': vocab}, agent

def load_dqn_model(path, env):
    """Load DQN model using existing function"""
    return load(path, env), None

parser = argparse.ArgumentParser()
parser.add_argument(
    '--env_key',
    default="MiniGrid-Empty-5x5-v0",
    help="Environment"
)
parser.add_argument(
    '--agent_type',
    default="dqn",
    choices=["dqn", "ppo"],
    help="Type of agent to load"
)
parser.add_argument(
    '--egocentric',
    default=True,
    help="Egocentric or allocentric",
    action='store_true'
)
parser.add_argument(
    '--exp',
    default=None,
    help="Task expression"
)
parser.add_argument(
    '--num_dists',
    type=int,
    default=1,
    help="Number of distractors"
)
parser.add_argument(
    '--size',
    type=int,
    default=7,
    help="Grid size"
)
parser.add_argument(
    '--obs_size',
    type=int,
    default=None,
    help="Observation size"
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=None
)
parser.add_argument(
    '--save',
    default=False,
    help="draw what the agent sees",
    action='store_true'
)
parser.add_argument(
    '--fps',
    type=int,
    default=3,
    help="FPS for GIF output"
)

args = parser.parse_args()

# Create environment
if "Custom" in args.env_key:
    env = gym.make(args.env_key, exp=args.exp, num_dists=args.num_dists, size=args.size, seed=args.seed)
else:
    env = gym.make(args.env_key)

# Apply wrappers
env = FullyObsWrapper(env, egocentric=args.egocentric)
env = RGBImgObsWrapper(env, tile_size=8)

def fig_image(fig, pad=0, h_pad=None, w_pad=None, rect=(0,0,1,1)):
    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def select_action_ppo(agent, obs, mission):
    """Select action using PPO agent"""
    action, _, _ = agent.get_action(obs, mission)
    return action

def select_action_ppo_deterministic(agent, obs, mission):
    """Select action deterministically for fair evaluation"""
    return agent.get_action_deterministic(obs, mission)

if __name__ == '__main__':
    # Set model path based on agent type
    path = f'modelsego/{args.env_key}_{args.agent_type}'
    
    if not os.path.exists(path):
        print(f"Model not found at {path}")
        print("Available models:")
        models_dir = 'modelsego/'
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('_dqn') or file.endswith('_ppo'):
                    print(f"  - {file}")
        exit(1)
    
    print('Loading model...')
    
    # Load model based on agent type
    if args.agent_type == 'ppo':
        model, agent = load_ppo_model(path, env)
    else:
        model, agent = load_dqn_model(path, env)

    if args.save:
        print('Saving video...')    
        images = []
    else:
        print('Visualizing...')     
        window = Window(args.env_key)   
    
    max_episodes = 4
    max_trajectory = 50
    
    success_count = 0
    total_reward = 0
    
    with torch.no_grad():
        for episode in range(max_episodes):
            obs = env.reset()
            mission = tokenize(obs['mission'], model['vocab'])
            
            if not args.save:
                window.set_caption(obs['mission'])
            
            episode_reward = 0
            done = False
            
            for step in range(max_trajectory):
                # Render current state
                if not args.save:
                    img = env.render('rgb_array')
                    window.show_img(img)
                else:
                    image_allocentric = env.render("rgb_array", highlight=True)
                    image_egocentric = obs["image"]

                    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Allocentric view
                    axs[0].set_title("Global View", fontsize=16)
                    axs[0].set_xticks([])
                    axs[0].set_yticks([])
                    axs[0].imshow(image_allocentric)
                    
                    # Egocentric view
                    axs[1].set_title("Agent's View", fontsize=16)
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    axs[1].imshow(image_egocentric)
                    
                    fig.suptitle(f"Mission: {obs['mission']}\nStep: {step}, Reward: {episode_reward}", 
                                fontsize=14, y=0.95)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])
                    
                    images.append(fig_image(fig))
                
                # Select action based on agent type
                if args.agent_type == 'ppo':
                    # action = select_action_ppo(agent, obs['image'], mission)
                    action = select_action_ppo_deterministic(agent, obs['image'], mission)
                    # action, _, _ = agent.get_action(obs['image'], mission)
                    # print(f"State shape: {obs['image'].shape}")
                    # print(f"Mission shape: {mission.shape}")
                else:
                    action = select_action(model, obs['image'], mission)
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Check if window closed
                if not args.save and (hasattr(window, 'closed') and window.closed):
                    break
                
                if done:
                    success_count += 1
                    break
            
            total_reward += episode_reward
            
            status = "SUCCESS" if done and reward > 0 else "FAILED"
            print(f"Episode {episode+1}: {status}, Steps: {step+1}, Reward: {episode_reward}")
            
            if not args.save and (hasattr(window, 'closed') and window.closed):
                break
    
    # Print summary
    print(f"\nSummary: {success_count}/{max_episodes} successful episodes")
    print(f"Average reward: {total_reward/max_episodes:.2f}")
    
    # Save GIF if requested
    if args.save and images:
        os.makedirs("images", exist_ok=True)
        output_path = f"images/trained_agent_{args.env_key}_{args.agent_type}.gif"
        imageio.mimsave(output_path, images, fps=args.fps, loop=0)
        print(f"GIF saved to {output_path}")