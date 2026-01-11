import os
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from codecarbon import track_emissions

# Importing environment and sensory data system modules
from pushpak_env import make_pushpak_env
from sensors import NoisySensorWrapper

# --- CONFIGURATION ---
MODEL_NAME = "pushpak_model"
MODELS_DIR = "Models"
LOG_DIR = "Logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_PATH = f"{MODELS_DIR}/{MODEL_NAME}"
STATS_PATH = f"{MODELS_DIR}/{MODEL_NAME}_vec_normalize.pkl"

# --- ENV CREATION FUNCTION ---
def create_env():
    raw_env = make_pushpak_env() # Raw Environment Creation
    env = NoisySensorWrapper(raw_env, noise_level=0.01) # Noise Creation
    env = FrameStackObservation(env, stack_size=4) # Frame Stacking
    env = FlattenObservation(env) # Flattening
    return env

# --- Carbon Tracking Decorator ---
@track_emissions(
    project_name="pushpak_lander_ppo",
    output_dir=LOG_DIR,
    save_to_file=True,
    experiment_id="v2_fixed_eval_v2"
)

# --- MAIN TRAINING FUNCTION ---
def main():
    print("🌍 CodeCarbon tracking enabled → Logs/emissions.csv")
    
    # 1. SETUP TRAINING ENVIRONMENT
    vec_env = make_vec_env(create_env, n_envs=8) # Vectoization 8 in parallel
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.) # VecNormalize Wrapper
    vec_env.training = True
    
    if os.path.exists(STATS_PATH):
        print("📈 Loading training env normalization stats...")
        vec_env = VecNormalize.load(STATS_PATH, vec_env)
    else:
        print("🆕 No stats found. Starting normalization from scratch.")

    # 2. MODEL SETUP
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"🧠 Loading existing model: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=vec_env)
    else:
        print("🐣 No model found. Creating a NEW brain.")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,      # ← SLOWER (was 3e-4)
            n_steps=2048,            # ← Standard
            batch_size=128,          # ← Larger, stable gradients
            n_epochs=5,              # ← Fewer (was 10, overfitting)
            gamma=0.99,              # ← Higher (was 0.995)
            gae_lambda=0.95,
            clip_range=0.1,          # ← Tighter (was 0.2)
            ent_coef=0.01,           # ← Higher exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
        )
        
    # Formula to get your n_steps right:
    # [1] rollout_size = n_env * n_steps
    # [2] n_steps = (rollout_size_target) / (number_of_envs)

    # For example, with 8 envs and a desired rollout size of 8192:
    # n_steps = 8192 / 8 = 1024

    # Recommended rollout sizes:
        # 2048 - minimum
        # 4096 - ideal
        # 8192 - best

    # where,
        # n_envs are recommended to have equal or less than your CPU or GPU cores.
        # If you can't go up above 4 environments, the formula would be:
        # n_steps = 8192 / 4 = 2048

    # 3. EVALUATION ENVIRONMENT
    print("🔧 Setting up evaluation environment...")
    eval_env = make_vec_env(create_env, n_envs=1)
    
    # ✅ CRITICAL: Exact same wrapper order as training env
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    if os.path.exists(STATS_PATH):
        print("📈 Loading eval env stats from:", STATS_PATH)
        eval_env = VecNormalize.load(STATS_PATH, eval_env)  # Matches training wrapper order
        eval_env.training = False
        eval_env.norm_reward = False  # 👈 Disable reward norm AFTER loading
        print("✅ Eval env: training=False, norm_reward=False")
    else:
        print("⚠️ No stats for eval - using fresh normalization")
        eval_env.training = False
        eval_env.norm_reward = False

    # Callbacks - EvalCallback (every 20k steps)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOG_DIR,
        eval_freq=20_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Callbacks - CheckpointCallback (every 500k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,  # Fewer checkpoints
        save_path=MODELS_DIR,
        name_prefix=MODEL_NAME + "_ckpt",
    )

    # 4. TRAINING LOOP
    TIMESTEPS = 5_000_000
    print(f"🚀 Training for {TIMESTEPS:,} timesteps...")
    print("📊 Run `python -m tensorboard.main --logdir Logs` to monitor!")
    
    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            callback=[eval_callback, checkpoint_callback],
        )
    except KeyboardInterrupt:
        print("⚠️ Training interrupted manually. Saving...")

    # 5. SAVE EVERYTHING
    print("💾 Saving Model and Stats...")
    model.save(MODEL_PATH)
    vec_env.save(STATS_PATH)
    print("✅ Done!")
    
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()