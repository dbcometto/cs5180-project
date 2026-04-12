# Marv Training Notes

PPO GAE

637.9s to 26
21483.1s to 760


## Setup

# PPO
epsilon = 0.2
beta = 0.02
lambda_gae = 0.90
gamma = 0.98
alpha_logit = 0.001
alpha_value = 0.001
do_normalize_advantage = True

# Training
resume_epoch = 28
checkpoint_interval = 250
batch_size = 64
optimizer_epochs = 8

# World
step_limit = 499
use_fixed_map = False
enable_extra_channels = True
do_smooth_complete_reward = True
do_smooth_end_dist = True

### Networks
logit_network = BetterConvNetwork(input_channels=obs_channels, output_width=action_dim, 
                 hidden_channels1=64, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=96, kernel2=3, stride2=1, padding2=1,
                 hidden_channels3=128, kernel3=3, stride3=1, padding3=1,
                 poolwidth = 8, poolheight = 8,
                 fc1_width = 128)
value_network = BetterConvNetwork(input_channels=obs_channels, output_width=1, 
                 hidden_channels1=64, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=96, kernel2=3, stride2=1, padding2=1,
                 hidden_channels3=128, kernel3=3, stride3=1, padding3=1,
                 poolwidth = 8, poolheight = 8,
                 fc1_width = 128)