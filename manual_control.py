"""
Control the environment with keyboard
"""

import argparse
import numpy as np
import gym
import envs.envs
from envs.wrappers import *
from gym_minigrid.window import Window

class ManualControl:
    def __init__(
        self,
        env,
        agent_view=False
    ) -> None:
        self.env = env
        self.agent_view = agent_view

        self.window = Window(args.env_key)
        self.window.reg_key_handler(self.key_handler)

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset()
        self.window.show(block=True)

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        print('reward=%.2f, action=%.2f' % (reward, action))

        if done:
            print("done!")
            self.reset()
        else:
            self.redraw(obs)

    def redraw(self, obs):
        if not self.agent_view:
            # Render the full environment
            img = self.env.render("rgb_array", highlight=False)
        else:
            # Show what the agent sees (processed observation)
            img = obs["image"]
        
            # Debug: ensure we have a valid image
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                print(f"Warning: Empty image received. Shape: {img.shape}")
                # Create a placeholder image
                img = np.zeros((84, 84, 3), dtype=np.uint8)
                img[40:44, 40:44] = 255  # Add a white square for visibility
    
        self.window.show_img(img)

    def reset(self):
        obs = self.env.reset()

        if hasattr(self.env, "mission"):
            print("Mission: %s" % self.env.mission)
            self.window.set_caption(self.env.mission)

        self.redraw(obs)

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.window.close()
            return
        if key == "backspace":
            self.reset()
            return
        if key == "v":
            # Toggle between agent view and environment view
            self.agent_view = not self.agent_view
            print(f"Agent view: {self.agent_view}")
            self.redraw(self.env._last_obs)
            return

        if "Custom" in args.env_key:
            key_to_action = {
                "left": self.env.task_actions.left,
                "right": self.env.task_actions.right,
                "up": self.env.task_actions.forward,
                "enter": self.env.task_actions.done,
            }
        else:
            key_to_action = {
                "left": self.env.actions.left,
                "right": self.env.actions.right,
                "up": self.env.actions.forward,
                " ": self.env.actions.toggle,
                "pageup": self.env.actions.pickup,
                "pagedown": self.env.actions.drop,
                "enter": self.env.actions.done,
            }
        if key in key_to_action:
            action = key_to_action[key]
            self.step(action)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--env_key',
    default="MiniGrid-Empty-5x5-v0",
    help="Environment"
)
parser.add_argument(
    '--observation_type',
    default="full_egocentric",
    choices=["full_egocentric", "full_allocentric", "partial_egocentric", "partial_allocentric"],
    help="Type of observation"
)
parser.add_argument(
    '--view_size',
    type=int,
    default=3,
    help="View size for partial observation"
)
parser.add_argument(
    '--exp',
    default=None,
    help="Task expression"
)
parser.add_argument(
    '--num_dists',
    type=int,
    default=9,
    help="Number of distractors"
)
parser.add_argument(
    '--size',
    type=int,
    default=12,
    help="Grid size"
)
parser.add_argument(
    '--obs_size',
    type=int,
    default=None,
    help="Observation size for rendering"
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=None
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw what the agent sees",
    action='store_true'
)

args = parser.parse_args()

# Create the base environment
if "Custom" in args.env_key:
    env = gym.make(args.env_key, exp=args.exp, num_dists=args.num_dists, size=args.size, seed=args.seed)
else:
    env = gym.make(args.env_key)

# Apply observation wrappers based on the selected type
if args.observation_type == "full_egocentric":
    env = FullyObsWrapper(env, egocentric=True)
    print("Using: Full Egocentric Observation")
elif args.observation_type == "full_allocentric":
    env = FullyObsWrapper(env, egocentric=False)
    print("Using: Full Allocentric Observation")
elif args.observation_type == "partial_egocentric":
    env = PartialObsWrapper(env, view_size=args.view_size, egocentric=True)
    print(f"Using: Partial Egocentric Observation (view size: {args.view_size})")
elif args.observation_type == "partial_allocentric":
    env = PartialObsWrapper(env, view_size=args.view_size, egocentric=False)
    print(f"Using: Partial Allocentric Observation (view size: {args.view_size})")

# Apply RGB wrapper for pixel observations
env = RGBImgObsWrapper(env, tile_size=32, obs_size=args.obs_size)

manual_control = ManualControl(env, agent_view=args.agent_view)
manual_control.start()