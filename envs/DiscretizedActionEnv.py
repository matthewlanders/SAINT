import gymnasium as gym
from gymnasium import spaces  
import numpy as np


class DiscretizedActionEnv(gym.Wrapper):
    def __init__(self, env, num_bins_per_action_dim):
        super().__init__(env)
        self.num_bins_per_action_dim = num_bins_per_action_dim

        if not isinstance(self.env.action_space, spaces.Box):
            raise ValueError("DiscretizedActionEnv only works with environments that have a continuous (Box) action space.")

        self.continuous_action_space = self.env.action_space
        self.num_continuous_actions = self.continuous_action_space.shape[0]

        nvec = [self.num_bins_per_action_dim] * self.num_continuous_actions
        self.action_space = spaces.MultiDiscrete(nvec)

        self.action_low = self.continuous_action_space.low
        self.action_high = self.continuous_action_space.high
        self.action_bin_widths = (self.action_high - self.action_low) / self.num_bins_per_action_dim
        self.observation_space = self.env.observation_space

        env_name = "Unknown Env"
        if hasattr(self.env, 'spec') and self.env.spec is not None:
            env_name = self.env.spec.id

        print(f"Wrapped {env_name}:")
        print(f"Original continuous action space: {self.continuous_action_space}")
        print(f"New discrete action space: {self.action_space}")
        print(f"Action space nvec: {self.action_space.nvec}")
        print(f"Observation space: {self.observation_space}")

    def _discrete_to_continuous_action(self, discrete_action):
        continuous_action = np.zeros_like(self.continuous_action_space.sample(), dtype=np.float32)
        for i in range(self.num_continuous_actions):
            
            bin_index = discrete_action[i]
            continuous_action[i] = self.action_low[i] + \
                                   self.action_bin_widths[i] * (bin_index + 0.5)
        
        return np.clip(continuous_action, self.action_low, self.action_high)

    def step(self, discrete_action):
        continuous_action = self._discrete_to_continuous_action(discrete_action)
        
        observation, reward, terminated, truncated, info = self.env.step(continuous_action)
        done = terminated or truncated  
        return observation, reward, done, info  

    def reset(self, **kwargs):
        seed = kwargs.pop('seed', None)  
        options = kwargs.pop('options', None)
        reset_args = {}

        if seed is not None:
            reset_args['seed'] = seed
        if options is not None:
            reset_args['options'] = options

        reset_args.update(kwargs)

        try:
            observation, info = self.env.reset(**reset_args)
        except TypeError as e:
            print(f"Warning: env.reset with {reset_args} failed ({e}). Trying simpler reset.")
            if seed is not None and not reset_args:  
                observation, info = self.env.reset(seed=seed)
            elif not reset_args:  
                observation, info = self.env.reset()
            else:  
                raise e  

        return observation  
