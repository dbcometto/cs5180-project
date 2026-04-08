# Bob Training Notes

6 Apr 2026:
Trained for 4256 batches of 5 episodes with 5 optimizer steps (9.5 hours, 34751.3s):
- logit_lr=0.001, value_lr = 0.001, entropy_bonus=0.02, do_normalize_advantage=False
- batch_size=5, optimizer_epochs=5, clip_epsilon=0.2, start_seed=2025, gamma=0.99, checkpoint_interval = 1000
Forgot to enable random maps...


train_env = TreeWorld(render_mode=None, step_limit=999, obs_as_tensor=True, use_fixed_map=True)


logit_network = SimpleConvNetwork(input_channels=obs_channels,output_width=action_dim)
value_network = SimpleConvNetwork(input_channels=obs_channels,output_width=1)
policy = DiscretePPO(logit_network,value_network,actions=action_list,logit_lr=0.001,value_lr = 0.001, entropy_bonus=0.02, do_normalize_advantage=False)
friend = Agent(policy)
friend.train(train_env,epochs=100_000,batch_size=5,optimizer_epochs=5,clip_epsilon=0.2,start_seed=2025,gamma=0.99, folderpath = folderpath, checkpoint_interval = 1000, resume_epoch=None)

friend.train(train_env,epochs=2,batch_size=5,optimizer_epochs=5,clip_epsilon=0.2,start_seed=2025,gamma=0.99, folderpath = folderpath, checkpoint_interval = 1000, resume_epoch=4256)