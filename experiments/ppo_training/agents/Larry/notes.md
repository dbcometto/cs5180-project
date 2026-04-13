# Larry Training Notes

Added scalar distance channel to hopefully help guide it back to the start
Dramatically reduced the network size.  Still larger than fred's.

Down to ~40s per batch to start, so already huge improvement

31189.5s to 640
50652.5s to 1457






## Setup

### Agent
agent_name = "Larry"
agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"

### PPO
epsilon = 0.2
beta = 0.05
lambda_gae = 0.95
gamma = 0.99
alpha_logit = 0.001
alpha_value = 0.001
do_normalize_advantage = True

### Resume
resume_epoch = None

### Training
checkpoint_interval = 100
batch_size = 32
optimizer_epochs = 4
minibatch_size = 512

### World
step_limit = 499
use_fixed_map = False
enable_extra_channels = True
enable_extra_dist_channel = True
do_smooth_complete_reward = True
do_smooth_end_dist = True
do_gate_ending = True

### Networks
logit_network = SimpleConvNetwork(input_channels=obs_channels, output_width=action_dim, 
                 hidden_channels1=32, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=64, kernel2=3, stride2=1, padding2=1,
                 poolwidth = 4, poolheight = 4,
                 fc1_width = 128)
value_network = SimpleConvNetwork(input_channels=obs_channels, output_width=1, 
                 hidden_channels1=32, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=64, kernel2=3, stride2=1, padding2=1,
                 poolwidth = 4, poolheight = 4,
                 fc1_width = 128)