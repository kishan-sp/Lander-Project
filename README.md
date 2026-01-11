# Quickstart (5 minutes)
```python
# 1. Clone & setup virtual environment (skip C++ tools if not on Windows) 
git clone <your-repo-url> 
cd pushpak-lander 
python -m venv venv 
venv\Scripts\activate # Windows 
# source venv/bin/activate # Linux/Mac 

# 2. Install dependencies (after C++ tools on Windows) 
pip install -r requirements.txt 
# 3. Train your first model (CPU: ~30min, GPU: ~10min) 
python train_agent.py 

# 4. Watch it land! 
python visualize_agent.py
```

# File Structure
```
pushpak-lander/
├── train_agent.py                    # Training script
├── visualize_agent.py                # Visualization script
├── requirements.txt                  # Dependencies
├── README.md                          # Setup instructions
├── PRETRAINED_MODELS.md              # Model information
├── download_pretrained_models.py     # Helper script
└── Models/                           # (Optional: can be downloaded)
    ├── pushpak_gnc_brain_v2.zip
    └── pushpak_gnc_brain_v2_vec_normalize.pkl
```
**Here is my File structure:**
<img width="529" height="591" alt="image" src="https://github.com/user-attachments/assets/508fe694-ab60-41d8-808f-4384f8967371" />
If you make changes in the file structure, make sure to change the file paths in the code too.

# Requirements
1. **Download:** Go to this link: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    - Click "Download Build Tools".

2. **Run the Installer:** It will update itself and open a window titled "Visual Studio Installer".

3. **The Critical Step (Don't miss this):**
    - You will see a list of "Workloads".
    - Check the box for **"Desktop development with C++"**.
    - _Note:_ This includes the "MSVC ... C++ x64/x86 build tools" and "Windows 10/11 SDK".
    - **Warning:** The download size is large (about **2 GB - 6 GB**). This will take time.

4. **Install:** Click "Install" and wait for it to finish.

5. **Restart:** Once finished, **Restart your Computer**. (This is mandatory to update your system Path).

6. After successfully installing Microsoft C++ Build tools then only download the following requirements.
```
stable-baselines3==2.2.1
gymnasium==0.29.1
codecarbon==4.1.2
tensorboard==2.14.1
numpy==1.24.3
torch==2.0.1
```

# Files and their uses
1. **sensors.py** - This file simulates that sensory readings that a rocket gets from its gyroscope, velocity, radar, accelerometers, etc. by creating a numpy array for 8 metrics mentioned in the sensors.py itself. After generating the numpy array, it is added with 5% of noise to simulate the real scenarios of a rocket.

2. **pushpak_env.py** - This file contains the environment in which the lander will be trained for landing. It contains features like gravity, wind, turbulence, wind_power, etc. I have user Gymnasium's Lunar-Lander-v3, if you have any better option please share it with me. 
   Anyways, it first initializes an environment, added some noise in the environment, finally the loop of action starts -> actions are observed, rewarded, and stored info -> then the model terminates or truncates according to the range of steps mentioned in the for loop (300) meaning it will truncate after 300 frames or 3 seconds and terminate means that lander crashed and episode ends.

3. **train_agent.py** - This is the main file which uses sensors and pushpak_env files for creating the model.
	   It contains the model's algorithm (Proximal Policy Optimization) which will be trained. The model has been tweaked according to my PC's limitations and here is the gist of what I changed:
   ```
	model = PPO(
	            "MlpPolicy",
	            vec_env,
	            verbose=1,
	            learning_rate=3e-4,
	            n_steps=2048,
	            batch_size=64,
	            n_epochs=10,
	            gamma=0.995,
	            gae_lambda=0.95,
	            clip_range=0.2,
	            vf_coef=0.5,
	            max_grad_norm=0.5,
	            ent_coef=0.005,
	            tensorboard_log=LOG_DIR,
	        )
   ```
	
	- **learning_rate:** Step size 3e-4 or 0.0003
	
	- **n_steps:** Steps after which the model will update its policy, a larger `n_steps` value means more data is collected before an update, which can lead to more stable updates but requires more memory and computational resources. The default value for PPO in SB3 is 2048.
	
	- **batch_size:** It is common practice to use batch sizes that are powers of two (e.g., 32, 64, 128, 256) because modern GPUs are optimized for these sizes, leading to better computational efficiency. A good approach is to start with a moderate, standard size (like 32 or 64) and test the model's performance while adjusting other parameters like the learning rate.
	  A larger batch provides stable and accurate gradient but can lead to *overfitting*.
	
	- **n_epochs:** When an entire dataset is passed forward and backward through the neural network Once, it is called 1 epoch.
		- The agent interacts with the environment for a fixed number of steps (`n_steps`) across all parallel environments to fill a "rollout buffer".
		- This collected data is divided into smaller "mini-batches" based on our defined `batch_size`.
		- For each epoch:
			- The agent performs a full pass through the buffer by iterating over every mini-batch.
			- At each mini-batch, it calculates the clipped surrogate loss and updates the neural network weights.
		- Once `n_epochs` passes are completed, the old data is cleared, and the agent returns to step 1 to collect fresh experience.
		- **Stability vs. Overfitting:**
			- **Higher `n_epochs` (e.g., 10):** Squeezes more information from collected data but risks "over-updating" the policy, which can cause performance to collapse if the agent deviates too far from the data it just collected.
			- **Lower `n_epochs` (e.g., 3–5):** Provides more stable, conservative updates but may require more total environment interactions to reach peak performance.
	
	- **eval_env:** During training, your agent only learns from `vec_env` (8 parallel envs). But you need to know:
		- **Is my agent actually getting better?**
		- **What's the _real_ performance without training noise?**
		- **Should I save this checkpoint as the "best" model?**
		- Without eval_env:  ```
		  model.learn(total_timesteps=3M)  # Train on vec_env 
		  No way to know if agent is improving!
		  Could be overfitting, unlucky, or genuinely learning```
		- With eval_env: ```
		  Every 20k steps:
		  Run 10 deterministic episodes on eval_env
		  → Compute mean return
		  → Save model if it's the best so far
		  → Log results to TensorBoard for monitoring```
		
		- Example Output: ```
		  Eval step 20000:  ep_rew_mean=125.4  ep_len_mean=287  (new best!)
		  Eval step 40000:  ep_rew_mean=142.1  ep_len_mean=305  (new best!)
		  Eval step 60000:  ep_rew_mean=139.8  ep_len_mean=301  (no improvement)```
		
		  - **NOTE:** This topic is something that I experienced hard time understanding, if someone understands it better please help me understand too. So to make my lander work I used Gemini for writing this code. I gave it my code and asked him to modify the code such that best model (best_model.zip) is recognized and presevered.
		  
		  - Practical Example:
		    Step 0: Training: vec_env runs 8 parallel episodes, agent learns
			  
			Step 20,000:
			  🔔 Eval triggered
			  - Reset eval_env (single clean environment)
			  - Run 10 episodes deterministically
			  - Mean reward = 42.5, Best so far: 10.2 ✅ NEW BEST!
			  - Save model → Models/best_model.zip
			  - Log to TensorBoard
			  
			Step 40,000:
			  🔔 Eval triggered
			  - Mean reward = 38.3, Best so far: 42.5 ❌ (didn't improve)
			  - Don't save model
			  
			Step 60,000:
			  🔔 Eval triggered
			  - Mean reward = 55.1, Best so far: 42.5 ✅ NEW BEST!
			  - Save model → Models/best_model.zip
	
	- **Timesteps:** Most agents (PPO/DQN) begin to stabilize around **400k–500k steps**. A full "solution" (consistent 200+ ep_rew_mean score) usually requires **1M+ steps** to handle edge cases like wind. I started at 100_000 steps and then after that I increased it to 3_000_000 steps so that model is trained properly.
	
	- Expected Performance (3M timesteps)```
		Time to convergence:    2-10 minutes (GPU) / 10-30 min (CPU)
		Energy used:            0.15-0.25 kWh
		CO₂ emissions:          0.1-0.18 kg CO₂e (like 500m-1km driving)
		
		Learning curve:
		  Steps 0-500k:         ep_rew_mean = -50 → 50 (random → decent)
		  Steps 500k-1.5M:      ep_rew_mean = 50 → 140 (fast improvement)
		  Steps 1.5M-3M:        ep_rew_mean = 140 → 180 (fine-tuning)
		  
		Final metrics:
		  ep_rew_mean:          170-190 (landed ~70% of time)
		  ep_len_mean:          250-350 (reasonable landing time)
		  explained_variance:   0.80-0.90 (good value prediction)
		  approx_kl:            0.008-0.015 (well-controlled)
		  value_loss:           15-30 (stable)```

# Troubleshooting
1. **Model not improving?**
```copy
# Reduce learning rate
learning_rate=1e-4,  # was 3e-4

# Increase training time
TIMESTEPS = 5_000_000,  # was 3M

# Check TensorBoard for diverging value_loss
```

2. **Out of Memory error?**
```copy
n_envs=4,  # was 8
n_steps=1024,  # was 2048
batch_size=32,  # was 64
```
