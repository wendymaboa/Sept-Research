#!/usr/bin/env python3
"""
Simple PPO test script to verify it works with different observation types
"""

import numpy as np
import torch
import gym
import gym_minigrid
import envs.envs
from envs.wrappers import *
from ppo_agent import PPOAgent
from library import tokenize
import matplotlib.pyplot as plt
import time

def simple_ppo_test(env_wrapper_type='full_allocentric', max_episodes=500, print_freq=50):
    """
    Test PPO on a simple environment with different observation wrappers
    
    Args:
        env_wrapper_type: 'full_allocentric', 'full_egocentric', 'partial_allocentric', 'partial_egocentric'
        max_episodes: Number of episodes to train
        print_freq: How often to print progress
    """
    print(f"\n=== Testing PPO with {env_wrapper_type} ===")
    
    # Create base environment - simple pickup task
    env = gym.make('Minigrid-PickUpObj-Custom-v0', 
                   exp="red & key",  # Simple goal: pick up red key
                   size=6)  # Small environment
    
    # Apply appropriate wrapper
    if env_wrapper_type == 'full_allocentric':
        env = FullyObsWrapper(env, egocentric=False)
    elif env_wrapper_type == 'full_egocentric':
        env = FullyObsWrapper(env, egocentric=True)
    elif env_wrapper_type == 'partial_allocentric':
        env = PartialObsWrapper(env, view_size=5, egocentric=False)
    elif env_wrapper_type == 'partial_egocentric':
        env = PartialObsWrapper(env, view_size=5, egocentric=True)
    else:
        raise ValueError(f"Unknown wrapper type: {env_wrapper_type}")
    
    env = RGBImgObsWrapper(env, tile_size=8, obs_size=64)  # Small image size for fast training
    
    # Create PPO agent
    agent = PPOAgent(env, 
                    learning_rate=3e-4,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_epsilon=0.2,
                    ppo_epochs=4,
                    batch_size=32)  # Smaller batch for simple test
    
    # Initialize simple vocabulary for mission
    agent.vocab = {'red': 1, 'key': 2, 'pick': 3, 'up': 4, 'a': 5, '&': 6}
    
    # Training tracking
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    
    start_time = time.time()
    
    for episode in range(max_episodes):
        obs = env.reset()
        mission = tokenize(obs['mission'], agent.vocab)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < 100:  # Max 100 steps per episode
            # Get action from PPO agent
            action, log_prob, value = agent.get_action(obs['image'], mission)
            
            # Take step
            new_obs, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs['image'], mission, action, log_prob, value, reward, done)
            
            # Update tracking
            obs = new_obs
            episode_reward += reward
            episode_length += 1
        
        # Update agent after episode
        agent.update()
        
        # Track episode results
        episode_rewards.append(episode_reward)
        episode_successes.append(1 if episode_reward > 0 else 0)
        episode_lengths.append(episode_length)
        
        # Print progress
        if episode % print_freq == 0 and episode > 0:
            recent_rewards = episode_rewards[-print_freq:]
            recent_successes = episode_successes[-print_freq:]
            recent_lengths = episode_lengths[-print_freq:]
            
            avg_reward = np.mean(recent_rewards)
            success_rate = np.mean(recent_successes)
            avg_length = np.mean(recent_lengths)
            
            elapsed_time = time.time() - start_time
            episodes_per_sec = episode / elapsed_time
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Success Rate: {success_rate:4.2f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Eps/sec: {episodes_per_sec:.1f}")
            
            # Check if we're learning (success rate > 0.8 for last 100 episodes)
            if len(episode_successes) >= 100:
                recent_success_rate = np.mean(episode_successes[-100:])
                if recent_success_rate > 0.8:
                    print(f"\n✅ SUCCESS! Achieved {recent_success_rate:.2f} success rate!")
                    break
    
    # Final results
    total_time = time.time() - start_time
    final_success_rate = np.mean(episode_successes[-100:]) if len(episode_successes) >= 100 else np.mean(episode_successes)
    
    print(f"\n=== {env_wrapper_type} Results ===")
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Final success rate: {final_success_rate:.3f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Episodes per second: {len(episode_rewards)/total_time:.1f}")
    
    return episode_rewards, episode_successes, episode_lengths

def run_all_tests():
    """Run PPO tests on all observation types"""
    results = {}
    
    test_types = [
        'full_allocentric',
        'full_egocentric', 
        'partial_allocentric',
        'partial_egocentric'
    ]
    
    for test_type in test_types:
        try:
            rewards, successes, lengths = simple_ppo_test(test_type, max_episodes=300)
            results[test_type] = {
                'rewards': rewards,
                'successes': successes, 
                'lengths': lengths
            }
            print(f"✅ {test_type} test completed successfully")
        except Exception as e:
            print(f"❌ {test_type} test failed: {str(e)}")
            results[test_type] = None
        
        print("-" * 60)
    
    # Plot comparison if matplotlib available
    try:
        plot_comparison(results)
    except:
        print("Could not create plots (matplotlib not available)")
    
    return results

def plot_comparison(results):
    """Plot learning curves for comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('PPO Learning Curves - Different Observation Types')
    
    for i, (test_type, data) in enumerate(results.items()):
        if data is None:
            continue
            
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot success rate (smoothed)
        successes = data['successes']
        window = min(50, len(successes) // 10)
        if window > 0:
            smoothed_success = np.convolve(successes, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(successes)), smoothed_success, label='Success Rate')
        
        ax.set_title(test_type.replace('_', ' ').title())
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Quick single test
    print("Starting PPO verification tests...")
    
    # Test 1: Full allocentric (should work easily)
    print("\n" + "="*60)
    simple_ppo_test('full_allocentric', max_episodes=200)
    
    # Test 2: Full egocentric (should work but maybe slower)
    print("\n" + "="*60) 
    simple_ppo_test('full_egocentric', max_episodes=200)
    
    # If you want to test all types, uncomment:
    # run_all_tests()