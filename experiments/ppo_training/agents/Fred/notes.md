# Fred Training Notes





Info:

train_env = TreeWorld(render_mode=None, step_limit=999, obs_as_tensor=True, use_fixed_map=False)

logit_network = SimpleConvNetwork(input_channels=obs_channels, output_width=action_dim)
value_network = SimpleConvNetwork(input_channels=obs_channels, output_width=1)
policy = DiscretePPO(logit_network, value_network, actions=action_list, 
                     logit_lr=0.001, value_lr = 0.001, entropy_bonus=0.02, 
                     do_normalize_advantage=False)
friend = Agent(policy)

friend.train(train_env,epochs=100_000, batch_size=32, optimizer_epochs=8, clip_epsilon=0.2, 
                start_seed=2025, gamma=0.99, 
                folderpath = folderpath, checkpoint_interval = 100, resume_epoch=None)


