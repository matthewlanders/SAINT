# SAINT: Attention-Based Modeling of Sub-Action Dependencies in Multi-Action Policies

This repository contains the official implementation of SAINT, an attention-based model for learning sub-action dependencies in multi-action reinforcement learning policies.

## Getting Started

SAINT and all baselines are implemented as single-file scripts to allow minimal setup and easy experimentation.

First, install the required dependencies:

```
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Code

### Running CityFlow

By default, the CityFlow agents run in the CityFlow-Irregular environment:

```
python agents/CityFlow/AR.py
python agents/CityFlow/Factored.py
python agents/CityFlow/PPO.py
python agents/CityFlow/SAINT.py
python agents/CityFlow/Wol-DDPG.py
```

To use the CityFlow-Linear environment, specify the corresponding config path:

```
python agents/CityFlow/SAINT.py --cityflow_config agents/cityflow/configs/Linear/config.json
```

### Running CoNE

When running an agent in CoNE for the first time, initialize the environment with:

```
load_terminal_states=False,
save_terminal_states=True,
```

For all subsequent runs, you can use the existing environment configuration:

```
python agents/CoNE/AR.py
python agents/CoNE/Factored.py
python agents/CoNE/PPO.py
python agents/CoNE/SAINT.py
python agents/CoNE/Wol-DDPG.py
```

### Running MuJoCo

By default, the MuJoCo agents run in the HalfCheetah-v4 environment:

```
python agents/CoNE/AR.py
python agents/CoNE/Factored.py
python agents/CoNE/SAINT.py
```

To use the Hopper or Walker2D environments, specify the corresponding gym env:

```
python agents/CoNE/SAINT.py --gym_env Hopper-v4
python agents/CoNE/SAINT.py --gym_env Walker2D-v4
```
