import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PPOAgent:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, ppo_epochs=4, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Initialize vocabulary (will be set externally)
        self.vocab = None
        
        # Get observation space dimensions
        obs_shape = env.observation_space.spaces['image'].shape
        
        self.actor_critic = ActorCritic(obs_shape, env.action_space.n)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Storage for episode data
        self.episode_data = {
            'states': [],
            'missions': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'log_probs': []
        }
    
    def get_action(self, state, mission):
        """Get action from current policy"""
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_tensor = state
                
        # Convert mission to tensor
        if isinstance(mission, (list, np.ndarray)):
            mission_tensor = torch.LongTensor(mission).unsqueeze(0)
        else:
            mission_tensor = mission
        
        # Remove debug prints for cleaner output
        # Ensure mission tensor has the right shape for embedding
        if mission_tensor.dim() == 3:
            if mission_tensor.shape[1] == 1:
                mission_tensor = mission_tensor.squeeze(1)
            else:
                mission_tensor = mission_tensor[:, 0, :]
        
        # Ensure correct device
        if next(self.actor_critic.parameters()).is_cuda:
            state_tensor = state_tensor.cuda()
            mission_tensor = mission_tensor.cuda()
        
        with torch.no_grad():
            action_probs, value = self.actor_critic(state_tensor, mission_tensor)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()
    
    def get_action_deterministic(self, state, mission):
        """Get deterministic action for evaluation (no sampling)"""
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_tensor = state
                
        if isinstance(mission, (list, np.ndarray)):
            mission_tensor = torch.LongTensor(mission).unsqueeze(0)
        else:
            mission_tensor = mission
        
        # Handle tensor shapes
        if mission_tensor.dim() == 3:
            if mission_tensor.shape[1] == 1:
                mission_tensor = mission_tensor.squeeze(1)
            else:
                mission_tensor = mission_tensor[:, 0, :]
        
        # Ensure correct device
        if next(self.actor_critic.parameters()).is_cuda:
            state_tensor = state_tensor.cuda()
            mission_tensor = mission_tensor.cuda()
        
        with torch.no_grad():
            action_probs, value = self.actor_critic(state_tensor, mission_tensor)
        
        # Take the most likely action instead of sampling
        action = action_probs.argmax(dim=-1)
        
        return action.item()    
    
    def store_transition(self, state, mission, action, log_prob, value, reward, done):
        """Store transition data"""
        self.episode_data['states'].append(state)
        self.episode_data['missions'].append(mission)
        self.episode_data['actions'].append(action)
        self.episode_data['log_probs'].append(log_prob)
        self.episode_data['values'].append(value)
        self.episode_data['rewards'].append(reward)
        self.episode_data['dones'].append(done)
    
    def compute_advantages_and_returns(self, next_value=0):
        """Compute GAE advantages and returns"""
        rewards = self.episode_data['rewards']
        values = self.episode_data['values']
        dones = self.episode_data['dones']
        
        advantages = []
        returns = []
        gae = 0
        
        # Convert values to float if they're tensors
        values_float = []
        for v in values:
            if torch.is_tensor(v):
                values_float.append(v.item())
            else:
                values_float.append(v)
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = values_float[step + 1]
            
            delta = rewards[step] + self.gamma * next_value_step * next_non_terminal - values_float[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values_float[step])
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO"""
        if len(self.episode_data['rewards']) == 0:
            return
        
        # For very short episodes, just clear and return
        if len(self.episode_data['rewards']) < 5:
            self.clear_memory()
            return
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages_and_returns()
        
        # Convert to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare batch data - Fix tensor stacking issues
        try:
            states = torch.stack([torch.FloatTensor(s) for s in self.episode_data['states']])
            missions = torch.stack([torch.LongTensor(m) for m in self.episode_data['missions']])
            actions = torch.tensor(self.episode_data['actions'])
            old_log_probs = torch.stack(self.episode_data['log_probs'])
        except Exception as e:
            print(f"Error preparing batch data: {e}")
            self.clear_memory()
            return
        
        # Move to GPU if available
        device = next(self.actor_critic.parameters()).device
        states = states.to(device)
        missions = missions.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        # PPO update epochs
        dataset_size = len(self.episode_data['rewards'])
        
        for epoch in range(self.ppo_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Skip if batch is too small
                if len(batch_indices) < 2:
                    continue
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_missions = missions[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                try:
                    # Forward pass
                    action_probs, values = self.actor_critic(batch_states, batch_missions)
                    dist = Categorical(action_probs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    
                    # PPO loss calculation
                    ratio = torch.exp(new_log_probs - batch_old_log_probs.detach())
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                    
                    # Total loss
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                    
                    # Check for NaN
                    if torch.isnan(loss):
                        print("Warning: NaN loss detected, skipping update")
                        continue
                    
                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                    self.optimizer.step()
                    
                except Exception as e:
                    print(f"Error in PPO update: {e}")
                    continue
        
        # Clear episode data
        self.clear_memory()
    
    def clear_memory(self):
        """Clear stored episode data"""
        for key in self.episode_data:
            self.episode_data[key] = []
    
    def save(self):
        """Save model"""
        if hasattr(self, 'path') and self.vocab is not None:
            torch.save({
                'model_state_dict': self.actor_critic.state_dict(),
                'vocab': self.vocab.vocab if hasattr(self.vocab, 'vocab') else self.vocab
            }, self.path)
            print(f"Model saved to {self.path}")


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, num_actions, mission_vocab_size=100, mission_embed_dim=32):
        super(ActorCritic, self).__init__()
        
        # Image processing (CNN) - Fix channel ordering
        # obs_shape is (H, W, C), we need (C, H, W) for conv2d
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # Adaptive pooling for consistent output size
            nn.Flatten()
        )
        
        # Calculate CNN output size
        cnn_output_size = 64 * 8 * 8  # After adaptive pooling
        
        # Mission processing - handle sequences
        self.mission_embedding = nn.Embedding(mission_vocab_size, mission_embed_dim)
        self.mission_lstm = nn.LSTM(mission_embed_dim, mission_embed_dim, batch_first=True)
        
        # Combined features
        combined_features = cnn_output_size + mission_embed_dim
        
        # Actor and critic heads
        self.actor = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, mission):
        # Process image - ensure correct channel ordering
        if state.dim() == 4 and state.shape[-1] == 3:  # NHWC format
            state = state.permute(0, 3, 1, 2)  # Convert to NCHW
        
        image_features = self.cnn(state)
        
        # Process mission sequence
        mission_embedded = self.mission_embedding(mission)
        
        # Use LSTM to process the sequence and get a single vector representation
        lstm_out, (hidden, cell) = self.mission_lstm(mission_embedded)
        mission_features = hidden[-1]  # Take the last hidden state
        
        # Combine features
        combined = torch.cat([image_features, mission_features], dim=1)
        
        # Get action probabilities and value
        action_probs = self.actor(combined)
        value = self.critic(combined)
        
        return action_probs, value