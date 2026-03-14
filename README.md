# CS-5180 RL Project

This repo contains code to create an RL environment, train agents on policies to act within them, and evaluate the performance of the agents.


## Install
To get started, (after activating a venv and adding it to the `.gitignore`) run
```bash
pip install -r requirements.txt
```
from the repository root.


## Description

The `treescan` folder is a python package for all the reusable code, the `experiments` folder has a bunch of one-off tests.

A few reusable objects, which are implemented to make things easier, are in the table below.

| Object | Description |
|---------------|-------------|
| Environment | Defines the environment, the action space, observation space, etc |
| Agent | A holder for a policy, handles training and testing |
| Policy | Implements a single algorithm, trains itself |
| Network | A specific neural network to be given to a policy that requires it for a single given agent |


## Testing 

The workflow so far has been to: 
1. Implement the algorithm(s) inside the `treescan` package
2. In an experiment folder, create: 
   - a training script to produce the agents and save them to a file 
   - a testing script to test the agents and save the results to files 
   - a graphing script to plot 
   - a demo script to render

## Structure

The repo structure is given below.

```text
Root/
├───experiments/
│   ├─── experiment1/
│   ├─── experiment2/
│   └─── ...
│
├───treescan/
│   ├───src/
│   │   └───treescan/
│   │       ├───agents/
│   │       ├───environments/
│   │       ├───networks/
│   │       ├───policies/
│   │       ├───utils/
│   │       └───wrappers/ (unused so far)
│   ├───pyproject.toml
│   └───setup.cfg
│
├───.gitignore
├───README.md
└───requirements.txt
```

