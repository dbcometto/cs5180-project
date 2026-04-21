# Ned Training Notes


47338.9s to 1356







## Setup 

agent_name = "Ned"

logit_network = SimpleConvNetwork(input_channels=obs_channels, output_width=action_dim)
value_network = SimpleConvNetwork(input_channels=obs_channels, output_width=1)
policy = DiscretePPOGAE(logit_network, value_network, actions=action_list, 
                     logit_lr=0.001, value_lr = 0.001, entropy_bonus=0.07,
                     do_normalize_advantage=False)
friend = Agent(policy)

friend.train(train_env,epochs=100_000, batch_size=32, optimizer_epochs=8, clip_epsilon=0.2, 
                 start_seed=seed, gamma=0.99, lambda_gae=0.95,
                 folderpath = folderpath, checkpoint_interval = 100, resume_epoch=None)