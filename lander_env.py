import gymnasium as gym
from sensors import NoisySensorWrapper

# Environment Initialization Function
def make_pushpak_env(render_mode=None): 
   # modes - human (opens a window to display the simulation), 
   # rgb_array (returns frames as nparray), 
   # none (disables rendering)

   env = gym.make(
      "LunarLander-v3",
      continuous = True,
      enable_wind = True,
      wind_power = 15.0,
      turbulence_power = 1.5,
      gravity = -10.0,
      render_mode = render_mode
      )
   return env

# Flight Loop
   # In Reinforcement Learning, the interaction loop always follows a specific pattern: 
   # Reset -> Action -> Step -> Repeat.

if __name__ == "__main__":
    # 1. Create the Physics World
    env = make_pushpak_env(render_mode="human")
    observation, info = env.reset()

    # 2. Add the Sensor Layer
    # We set noise_level=0.1 (High noise) just so you can SEE it happening
    env = NoisySensorWrapper(env, noise_level=0.1)

    # 3. Reset (This now goes through the wrapper!)
    observation, info = env.reset()
    # reset() - Modifies the env after calling reset, 
    #    returning a modified observation using self.observation.

    print("🚀 Simulation Started! Watch the popup window...")
    
    for _ in range(300):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Readout: {observation[0]:.2f}")

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print("✅ Simulation finished successfully.")
