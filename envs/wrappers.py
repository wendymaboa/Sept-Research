"""
Useful Wrappers - Complete version with Partial Observations
"""

import gym
from gym import spaces
import numpy as np
from collections import deque
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, Goal

# Try to import cv2, but provide fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Using numpy fallback for resizing.")

def numpy_resize(image, target_size):
    """Simple numpy-based resize fallback when cv2 is not available"""
    if image.shape[:2] == target_size:
        return image
    
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Simple nearest-neighbor resize
    y_indices = np.linspace(0, h-1, target_h).astype(int)
    x_indices = np.linspace(0, w-1, target_w).astype(int)
    
    if len(image.shape) == 3:
        return image[np.ix_(y_indices, x_indices)].copy()
    else:
        return image[np.ix_(y_indices, x_indices)]


class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
        Default: Regular topdown view
        Optional: Egocentric topdown view
    """

    def __init__(self, env, egocentric=True):
        super().__init__(env)

        self.egocentric = egocentric
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )

    def observation(self, obs):        
        env = self.unwrapped
        full_grid = env.grid.encode()     

        if not self.egocentric:
            rgb_img = full_grid
            y, x = self.agent_pos
            rgb_img[y, x, :] = (OBJECT_TO_IDX["agent"], 0, self.agent_dir) 
        else:       
            s = full_grid.shape[0]
            y, x = self.agent_pos

            # Egocentric rotation
            agent_pos = full_grid[:,:,0]*0
            agent_pos[y,x] = 1
            k = 3 - self.agent_dir
            agent_pos = np.rot90(agent_pos, k=k)
            for i in range(3):
                full_grid[:,:,i] = np.rot90(full_grid[:,:,i], k=k)        
            x, y = np.where(agent_pos==1)
            x, y = x[0], y[0]

            # Egocentric position
            ox = s//2-x    
            rgb_img = full_grid.copy()
            if ox>=0:
                rgb_img[ox:s//2,:,:] = full_grid[:x,:,:]    
                rgb_img[s//2:,:,:] = full_grid[x:x+s//2+s%2,:,:]   
                rgb_img[:ox,:,:] = full_grid[x+s//2+s%2:,:,:]   
            else:
                ox = s+ox
                rgb_img[s//2:ox,:,:] = full_grid[x:,:,:]    
                rgb_img[:s//2,:,:] = full_grid[x-s//2:x,:,:]   
                rgb_img[ox:,:,:] = full_grid[:x-s//2,:,:]    
            full_grid = rgb_img.copy()
            rgb_img[:,s-(y+1):,:] = full_grid[:,:y+1,:] 
            rgb_img[:,:s-(y+1),:] = full_grid[:,y+1:,:] 

        return {
            'mission': obs['mission'],
            'image': rgb_img
        }


class PartialObsWrapper(gym.core.ObservationWrapper):
    """
    Partially observable wrapper with configurable view size
    Supports both egocentric and allocentric partial views
    Compatible with existing FullyObsWrapper system
    """
    
    def __init__(self, env, view_size=4, egocentric=True):
        super().__init__(env)
        self.view_size = view_size
        self.egocentric = egocentric
        
        # Update observation space - both modes now return only the partial view
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=255,
            shape=(view_size, view_size, 3),
            dtype='uint8'
        )
    
    def observation(self, obs):
        if self.egocentric:
            return self._egocentric_partial_obs(obs)
        else:
            return self._allocentric_partial_obs(obs)
    
    def _egocentric_partial_obs(self, obs):
        """
        Egocentric partial observation - applies full egocentric transformation then crops
        This maintains compatibility with your existing egocentric system
        """
        # First get the full egocentric view (exactly like FullyObsWrapper does)
        env = self.unwrapped
        full_grid = env.grid.encode()
        
        s = full_grid.shape[0]
        y, x = env.agent_pos
        agent_dir = env.agent_dir

        # Egocentric rotation (same as FullyObsWrapper)
        agent_pos = full_grid[:,:,0]*0
        agent_pos[y,x] = 1
        k = 3 - agent_dir
        agent_pos = np.rot90(agent_pos, k=k)
        for i in range(3):
            full_grid[:,:,i] = np.rot90(full_grid[:,:,i], k=k)        
        x, y = np.where(agent_pos==1)
        x, y = x[0], y[0]

        # Egocentric position (same as FullyObsWrapper)
        ox = s//2-x    
        rgb_img = full_grid.copy()
        if ox>=0:
            rgb_img[ox:s//2,:,:] = full_grid[:x,:,:]    
            rgb_img[s//2:,:,:] = full_grid[x:x+s//2+s%2,:,:]   
            rgb_img[:ox,:,:] = full_grid[x+s//2+s%2:,:,:]   
        else:
            ox = s+ox
            rgb_img[s//2:ox,:,:] = full_grid[x:,:,:]    
            rgb_img[:s//2,:,:] = full_grid[x-s//2:x,:,:]   
            rgb_img[ox:,:,:] = full_grid[:x-s//2,:,:]    
        full_grid = rgb_img.copy()
        rgb_img[:,s-(y+1):,:] = full_grid[:,:y+1,:] 
        rgb_img[:,:s-(y+1),:] = full_grid[:,y+1:,:] 

        # Add agent marker (same as FullyObsWrapper)
        center_y, center_x = s//2, s-1  # Agent is at bottom center in egocentric view
        rgb_img[center_y, center_x, :] = (OBJECT_TO_IDX["agent"], 0, 3)  # Agent facing up

        # Now crop around the agent position to get partial view
        half_view = self.view_size // 2
        
        # Calculate crop boundaries (agent is at center_y, center_x)
        
        # start_y = center_y - (self.view_size - 1)  # Agent at bottom of partial view
        # start_x = center_x - (self.view_size // 2)  # Agent centered in partial view
        start_y = center_y - half_view
        end_y = start_y + self.view_size
        start_x = center_x - half_view
        end_x = start_x + self.view_size
        
        # Create partial view with padding if needed
        partial_view = np.zeros((self.view_size, self.view_size, 3), dtype=np.uint8)
        
        # Copy the valid region
        for py in range(self.view_size):
            for px in range(self.view_size):
                full_y = start_y + py
                full_x = start_x + px
                if 0 <= full_y < s and 0 <= full_x < s:
                    partial_view[py, px] = rgb_img[full_y, full_x]
                # else remains zeros (empty space)

        return {
            'mission': obs['mission'],
            'image': partial_view
        }
    
    def _allocentric_partial_obs(self, obs):
        """Allocentric partial observation (top-down view of limited area around agent)"""
        env = self.unwrapped
        full_grid = env.grid.encode()
        
        # Get agent position
        y, x = env.agent_pos
        
        # Extract partial view area around agent
        half_view = self.view_size // 2
        partial_view = np.zeros((self.view_size, self.view_size, 3), dtype=np.uint8)
        
        for i in range(-half_view, half_view + 1):
            for j in range(-half_view, half_view + 1):
                world_x, world_y = x + j, y + i
                view_x, view_y = j + half_view, i + half_view
                
                # Check if within world bounds
                if (0 <= world_x < env.width and 
                    0 <= world_y < env.height and
                    0 <= view_x < self.view_size and
                    0 <= view_y < self.view_size):
                    partial_view[view_y, view_x] = full_grid[world_y, world_x]
        
        # Mark agent position
        center = self.view_size // 2
        partial_view[center, center, :] = (OBJECT_TO_IDX["agent"], 0, env.agent_dir)
        
        return {
            'mission': obs['mission'],
            'image': partial_view
        }


class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.
    Enhanced version that properly handles both full and partial observations.
    """

    def __init__(self, env, tile_size=8, obs_size=None):
        super().__init__(env)

        self.tile_size = tile_size
        self.obs_size = obs_size
        
        # Check if we have a partial observation wrapper in the stack
        self.is_partial = False
        self.is_egocentric = False
        
        # Walk through wrapper stack to find observation type
        current_env = env
        while hasattr(current_env, 'env'):
            if isinstance(current_env, PartialObsWrapper):
                self.is_partial = True
                self.is_egocentric = current_env.egocentric
                self.view_size = current_env.view_size
                break
            elif isinstance(current_env, FullyObsWrapper):
                self.is_egocentric = current_env.egocentric
                break
            current_env = current_env.env

        obs_shape = env.observation_space.spaces['image'].shape
        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype='uint8'
        )
        if self.obs_size:
            self.observation_space.spaces['image'] = spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_size, self.obs_size, 3),
                dtype='uint8'
            )

    def observation(self, obs):
        env = self.unwrapped
        
        if self.is_partial:
            # For partial observations, render the partial grid directly
            rgb_img = self._render_partial_grid(obs['image'])
        else:
            # For full observations, use the original rendering logic
            if self.is_egocentric:
                rgb_img = self._render_egocentric_full_grid(obs['image'])
            else:
                rgb_img = env.render(
                    mode='rgb_array',
                    highlight=False,
                    tile_size=self.tile_size
                )
        
        # Resize image
        if self.obs_size:
            if CV2_AVAILABLE:
                rgb_img = cv2.resize(rgb_img, (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA)
            else:
                rgb_img = numpy_resize(rgb_img, (self.obs_size, self.obs_size))

        return {
            'mission': obs['mission'],
            'image': rgb_img
        }
    
    def _render_partial_grid(self, encoded_grid):
        """Render a partial grid from encoded representation using proper MiniGrid rendering"""
        from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
        
        # Create a temporary grid from the encoded representation
        height, width = encoded_grid.shape[:2]
        grid = Grid(width, height)
        
        # Find agent position and remove it from the encoded grid for decoding
        agent_pos = None
        agent_dir = 3
        clean_encoded_grid = encoded_grid.copy()
        
        for y in range(height):
            for x in range(width):
                if encoded_grid[y, x, 0] == OBJECT_TO_IDX["agent"]:
                    agent_pos = (x, y)
                    agent_dir = encoded_grid[y, x, 2]
                    # Replace agent cell with empty space for decoding
                    clean_encoded_grid[y, x] = [0, 0, 0]
        
        # If no agent position was found, use center for partial views
        if agent_pos is None:
            agent_pos = (width // 2, height // 2)
        
        # Decode the cleaned grid (without agent) back to actual objects
        grid, _ = grid.decode(clean_encoded_grid)
        
        # Render using the same method as full observations
        rgb_img = grid.render(
            self.tile_size,
            agent_pos=agent_pos,
            agent_dir=agent_dir
        )
        
        return rgb_img
    
    def _render_egocentric_full_grid(self, encoded_grid):
        """Render egocentric full grid (for backward compatibility)"""
        from gym_minigrid.minigrid import Grid
        
        # For egocentric full view, agent should be at bottom center
        height, width = encoded_grid.shape[:2]
        
        # Remove agent from grid for decoding
        clean_encoded_grid = encoded_grid.copy()
        agent_pos = (width // 2, height - 1)  # Bottom center
        agent_dir = 3  # Facing up in egocentric view
        
        for y in range(height):
            for x in range(width):
                if encoded_grid[y, x, 0] == OBJECT_TO_IDX["agent"]:
                    clean_encoded_grid[y, x] = [0, 0, 0]
        
        # Decode and render
        grid = Grid(width, height)
        grid, _ = grid.decode(clean_encoded_grid)
        rgb_img = grid.render(
            self.tile_size,
            agent_pos=agent_pos,
            agent_dir=agent_dir
        )
        
        return rgb_img


# Helper functions for creating environments with different observation types
def create_fully_observable_env(env_key, egocentric=True, tile_size=8, obs_size=None, **env_kwargs):
    """
    Create environment with full observations (your existing system)
    """
    if "Custom" in env_key:
        env = gym.make(env_key, **env_kwargs)
    else:
        env = gym.make(env_key)
    
    env = FullyObsWrapper(env, egocentric=egocentric)
    env = RGBImgObsWrapper(env, tile_size=tile_size, obs_size=obs_size)
    
    return env


def create_partially_observable_env(env_key, egocentric=True, view_size=7, tile_size=8, obs_size=None, **env_kwargs):
    """
    Create environment with partial observations (new functionality)
    """
    if "Custom" in env_key:
        env = gym.make(env_key, **env_kwargs)
    else:
        env = gym.make(env_key)
    
    env = PartialObsWrapper(env, view_size=view_size, egocentric=egocentric)
    env = RGBImgObsWrapper(env, tile_size=tile_size, obs_size=obs_size)
    
    return env


def create_env_by_type(observation_type, env_key, **kwargs):
    """
    Factory function to create environments based on observation type string
    Useful for manual_control.py and other scripts
    """
    if observation_type == "full_egocentric" or observation_type == "egocentric_fully_observable":
        return create_fully_observable_env(env_key, egocentric=True, **kwargs)
    elif observation_type == "full_allocentric" or observation_type == "allocentric_fully_observable":
        return create_fully_observable_env(env_key, egocentric=False, **kwargs)
    elif observation_type == "partial_egocentric" or observation_type == "egocentric_partially_observable":
        view_size = kwargs.pop('agent_view_size', kwargs.pop('view_size', 7))
        return create_partially_observable_env(env_key, egocentric=True, view_size=view_size, **kwargs)
    elif observation_type == "partial_allocentric" or observation_type == "allocentric_partially_observable":
        view_size = kwargs.pop('agent_view_size', kwargs.pop('view_size', 7))
        return create_partially_observable_env(env_key, egocentric=False, view_size=view_size, **kwargs)
    else:
        raise ValueError(f"Unknown observation type: {observation_type}")


# Test the wrappers if run directly
if __name__ == "__main__":
    print("Testing wrapper combinations...")
    
    # Test 1: Full egocentric (your existing system)
    print("\n1. Testing Full Egocentric:")
    env = create_fully_observable_env("MiniGrid-Empty-5x5-v0", egocentric=True, tile_size=8)
    obs = env.reset()
    print(f"   Observation shape: {obs['image'].shape}")
    env.close()
    
    # Test 2: Full allocentric
    print("\n2. Testing Full Allocentric:")
    env = create_fully_observable_env("MiniGrid-Empty-5x5-v0", egocentric=False, tile_size=8)
    obs = env.reset()
    print(f"   Observation shape: {obs['image'].shape}")
    env.close()
    
    # Test 3: Partial egocentric (new)
    print("\n3. Testing Partial Egocentric:")
    env = create_partially_observable_env("MiniGrid-Empty-5x5-v0", egocentric=True, view_size=3, tile_size=8)
    obs = env.reset()
    print(f"   Observation shape: {obs['image'].shape}")
    env.close()
    
    # Test 4: Partial allocentric (new)
    print("\n4. Testing Partial Allocentric:")
    env = create_partially_observable_env("MiniGrid-Empty-5x5-v0", egocentric=False, view_size=3, tile_size=8)
    obs = env.reset()
    print(f"   Observation shape: {obs['image'].shape}")
    env.close()
    
    # Test 5: Factory function
    print("\n5. Testing Factory Function:")
    env = create_env_by_type("egocentric_partially_observable", "MiniGrid-Empty-5x5-v0", agent_view_size=3)
    obs = env.reset()
    print(f"   Factory-created env observation shape: {obs['image'].shape}")
    env.close()
    
    print("\n✓ All wrapper tests completed successfully!")