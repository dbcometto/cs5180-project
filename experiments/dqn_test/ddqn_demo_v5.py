"""Visualize a trained DDQN agent on its training env."""                                                                            
import torch                                                                                                                         
from treescan.agents import Agent
from version_configs import makeEnv                                                                             
                  
version = "v5"                                                                                                                       
agent_name = "ddqn_v5"   # or "dqn_v5"
seed = 2025                                                                                                                          
                                                                                                                                       
agents_folder = "/Users/adamlewis/Desktop/Northeastern/Reinforcement Learning/Project/cs5180-project/experiments/dqn_test/agents"    
agent = Agent.load(f"{agents_folder}/{agent_name}")                                                                                  
                                                                                                                                       
# build the env with V5's exact settings, but in human-render mode                                                                   
demo_env = makeEnv(version, force_fixed_map=True)                                                                                    
demo_env.render_mode = "human"                                                                                                       
demo_env.step_limit = 200  # matches V5
                                                                                                                                       
torch.manual_seed(seed)
agent.generate_trajectory(demo_env, seed=seed)                                                                                       
print("Finished")
demo_env.hold_frame(duration=5000)   