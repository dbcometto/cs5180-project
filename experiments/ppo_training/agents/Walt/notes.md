# Walt Notes

First few attempts: memory issues and couldn't get through the optimization
-> tried to clear up the tensor memory growth, and reduced batch size to 32

Finally:
    1160.2s to 4
    35906.9s to 106 (wow... maybe the network is too big...)




# Setup

### Agent
agent_name = "Walt"
agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"

### PPO
epsilon = 0.2
beta = 0.005
lambda_gae = 0.95
gamma = 0.99
alpha_logit = 0.0007
alpha_value = 0.0007
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
do_smooth_complete_reward = True
do_smooth_end_dist = True
do_gate_ending = True



### Networks
logit_network = BetterConvNetwork2(input_channels=obs_channels, output_width=action_dim, 
                 hidden_channels1=64, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=96, kernel2=3, stride2=1, padding2=1,
                 hidden_channels3=128, kernel3=3, stride3=1, padding3=1,
                 poolwidth = 8, poolheight = 8,
                 fc1_width = 128)
value_network = BetterConvNetwork2(input_channels=obs_channels, output_width=1, 
                 hidden_channels1=64, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=96, kernel2=3, stride2=1, padding2=1,
                 hidden_channels3=128, kernel3=3, stride3=1, padding3=1,
                 poolwidth = 8, poolheight = 8,
                 fc1_width = 128)