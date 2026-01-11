# This keeps your "Sensor Logic" separate from your "Physics or Environment Logic."

import gymnasium as gym
import numpy as np

class NoisySensorWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_level=0.05): # <-> 0.05 means about 5% error in readings
        # Initialize the standard wrapper
        super().__init__(env)
        self.noise_level = noise_level

    def observation(self, obs): # <-> Part of gymnasium's lunar-lander environment
        # 'obs' is the perfect array of 8 numbers coming from the reset() and step() function of pushpak_env.
        # The 8 numbers represent:
        # [0] Position X
        # [1] Position Y
        # [2] Velocity X
        # [3] Velocity Y
        # [4] Angle
        # [5] Angular Velocity
        # [6] Left Leg Contact
        # [7] Right Leg Contact
        # We will modify it here before returning it.
        
        return self.add_navic_noise(obs)

    def add_navic_noise(self, obs):

        # We will write the math logic here in the next step
        # Copy the observation so we don't break the original physics
        noisy_obs = obs.copy()
        
        # Generate random noise for X and Y
        # mean=0 (centered on truth), std=self.noise_level (spread of error)
        # noise_level is set to 0.05 previously, meaning ~5% error
        # loc is mean, scale is standard deviation
        x_noise = np.random.normal(loc=0, scale=self.noise_level)
        y_noise = np.random.normal(loc=0, scale=self.noise_level)
        
        # Add the noise to the Position variables (Index 0 and 1)
        noisy_obs[0] += x_noise  # Corrupt X
        noisy_obs[1] += y_noise  # Corrupt Y
        
        return noisy_obs