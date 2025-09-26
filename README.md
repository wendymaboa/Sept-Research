# dqn_instruction_following
Simple deep Q-learning implemention for instruction following. 

# Create conda environment
```
conda create --name minigrid python==3.7.11
```
```
pip install -r requirements.txt
```

# Egocentric vs Allocentric observations

```
python manual_control.py --env_key "Minigrid-PickUpObj-Custom-v0" --egocentric --size 12
```

<table>
  <tr>
    <td> <img src="images/forward_agent.gif"  alt="forward agent" width = 100% height = auto > </td>
    <td> <img src="images/rotating_agent.gif"  alt="rotating agent" width = 100% height = auto > </td>
    <td> <img src="images/random_agent.gif"  alt="random agent" width = 100% height = auto > </td>
  </tr>   
</table>

# Training agent and visualising learned policies

```
python train.py --env_key "MiniGrid-Empty-Random-5x5-v0" --egocentric

for allocentric

python train.py --agent_type ppo --env_key "MiniGrid-Empty-5x5-v0" --save_model


for egocentric 

python train.py --agent_type ppo --env_key "MiniGrid-Empty-5x5-v0" --save_model --egocentric

python train.py --agent_type ppo --env_key "MiniGrid-Empty-5x5-v0" --save_model --view_size 7 --partial_egocentric
```
```
python visualise.py --env_key "MiniGrid-Empty-Random-5x5-v0" --egocentric
python visualise.py --env_key "Minigrid-PickUpObj-Custom-v0" --egocentric


# Visualize DQN agent
python visualize.py --env_key "MiniGrid-Empty-5x5-v0" --agent_type dqn

# Save as GIF
python visualize.py --env_key "MiniGrid-Empty-5x5-v0" --agent_type dqn --save --fps 5

python visualise.py --env_key "MiniGrid-Empty-5x5-v0" --agent_type dqn --egocentric --save

# Visualize PPO agent  
python visualize.py --env_key "MiniGrid-Empty-5x5-v0" --agent_type ppo


# FOR MANUAL CONTROL
python manual_control.py --observation_type partial_egocentric --agent_view

```
<table>
  <tr>
    <td> <img src="images//trained_agent_MiniGrid-Empty-Random-5x5-v0.gif"  alt="trained_agent MiniGrid-Empty-Random-5x5-v0" width = 100% height = auto > </td>
    <td> <img src="images/trained_agent_Minigrid-PickUpObj-Custom-v0.gif"  alt="trained_agent MiniGrid-PickUpObj-Custom-v0" width = 100% height = auto > </td>
  </tr>   
</table>
