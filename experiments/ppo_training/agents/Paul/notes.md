# Paul Training Notes

Goal is to be like Fred but with GAE going as well

9 hrs 10 15406.  Almost has learned but still just quits :(






## Setup:
```python
logit_network = SimpleConvNetwork(input_channels=obs_channels, output_width=action_dim)
value_network = SimpleConvNetwork(input_channels=obs_channels, output_width=1)
policy = DiscretePPOGAE(logit_network, value_network, actions=action_list, 
                     logit_lr=0.001, value_lr = 0.001, entropy_bonus=0.02,
                     do_normalize_advantage=False)




friend.train(train_env,epochs=100_000, batch_size=32, optimizer_epochs=8, clip_epsilon=0.2, 
                 start_seed=2025, gamma=0.99, lambda_gae=0.95,
                 folderpath = folderpath, checkpoint_interval = 100, resume_epoch=None)
```