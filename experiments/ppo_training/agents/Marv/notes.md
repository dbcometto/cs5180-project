# Marv Training Notes

PPO GAE

Attempt 1: 
    33274.8s to 1569... and he just learned to quit :(

    Added a prevent early termination flag
    Going to add an entropy bonus curriculum 
    Will also increase the failure/success rewards to make them more important


Attempt 2:
    6056.2s to 228

    Forgot to enable advantage normalization :(



## New Setup

# PPO
epsilon = 0.2
beta = 0.01
lambda_gae = 0.95
gamma = 0.99
alpha_logit = 0.0005
alpha_value = 0.0005

# Training
resume_epoch = 2
checkpoint_interval = 250
batch_size = 64
optimizer_epochs = 8

# World
step_limit = 499
use_fixed_map = False
enable_extra_channels = True

# Overwriting Rewards
NEW_FAIL_REWARD = -50
NEW_SUCCESS_REWARD = 50












## Original Setup

### PPO
epsilon = 0.2
beta = 0.01
lambda_gae = 0.95
gamma = 0.99
alpha_logit = 0.0005
alpha_value = 0.0005

### Training
resume_epoch = 2
checkpoint_interval = 250
batch_size = 64
optimizer_epochs = 8

### World
step_limit = 999
use_fixed_map = False
enable_extra_channels = True


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