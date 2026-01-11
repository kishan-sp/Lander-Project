import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import FrameStackObservation, FlattenObservation

from pushpak_env import make_pushpak_env
from sensors import NoisySensorWrapper


MODEL_NAME = "pushpak_model"
MODELS_DIR = "Models"
MODEL_PATH = f"{MODELS_DIR}/{MODEL_NAME}.zip"
STATS_PATH = f"{MODELS_DIR}/{MODEL_NAME}_vec_normalize.pkl"


def create_viz_env():
    def make_env():
        raw = make_pushpak_env(render_mode="human")
        noisy = NoisySensorWrapper(raw, noise_level=0.01)
        stacked = FrameStackObservation(noisy, stack_size=4)
        flat = FlattenObservation(stacked)
        return flat
    return DummyVecEnv([make_env])


# Verify files
if not all(os.path.exists(p) for p in [MODEL_PATH, STATS_PATH]):
    raise FileNotFoundError("Missing model.zip or stats.pkl")

print("✅ Files OK")

# Load env + EXACT stats
env = create_viz_env()
env = VecNormalize.load(STATS_PATH, env)
env.training = False
env.norm_reward = False

# FIXED Stats print (correct VecNormalize attributes)
print("📊 Stats loaded:")
print(f"Obs count: {env.obs_rms.count}")
print(f"Obs var:  {np.mean(env.obs_rms.var):.3f}")

# Load model
model = PPO.load(MODEL_PATH, env=env)
print("🧠 Model ready")

print("\n🚀 PUSHPAK LANDER - SCIENTIST TEST")
print("Window shows trained performance. Close when done.\n")

obs = env.reset()
episode_reward = 0
episode_count = 0

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        episode_reward += rewards[0]
        
        if dones[0]:
            episode_count += 1
            status = "✅ PERFECT!" if episode_reward > 200 else "❌ Needs more training"
            print(f"📍 Ep {episode_count}: {episode_reward:.0f}pts {status}")
            episode_reward = 0
            obs = env.reset()
            
except KeyboardInterrupt:
    print("\n🛑 Demo stopped")

env.close()
print("✅ Test complete")
