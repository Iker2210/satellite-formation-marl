import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from env.satellite_env import SatelliteEnv

# --- CONFIGURACIÓN DE DISPOSITIVO ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Forzar CPU si es necesario

# --- RUTAS ---
# Nueva carpeta para la versión con "ojos nuevos" (Velocidades en el estado)
# Update paths before running
LOG_DIR = "logs/robot_4/ppo_simple_59"
MODEL_DIR = "models/multi_agent/robot_4/ppo_simple_59"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- HIPERPARÁMETROS ENTRENAMIENTO DESDE CERO ---
TOTAL_TIMESTEPS = 15_000_000  # Empezando de 0 necesitaremos algo más de tiempo
LEARNING_RATE = 3e-4         # Volvemos al LR estándar de PPO para aprendizaje rápido
SEED = 42
EVAL_FREQ = 50_000         
ENV_MAX_STEPS = 2000         # Mantener 2000 para dar tiempo a maniobras de esquiva

def make_env(seed: int):
    # El entorno DEBE tener el nuevo _get_state con las velocidades
    env = SatelliteEnv(max_steps=ENV_MAX_STEPS, verbose=False, seed=seed, n_agents=4)
    env = Monitor(env)
    return env

# Crear entornos
train_env = make_env(SEED)
eval_env = make_env(SEED + 2000)

# --- CREAR MODELO NUEVO (DESDE CERO) ---
print("Creando modelo PPO desde cero con el nuevo espacio de estados...")

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=LEARNING_RATE,
    n_steps=4096,           # Horizonte largo para coordinar a 4 agentes
    batch_size=128,         # Tamaño de batch equilibrado
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,         # Clip estándar para entrenamiento inicial
    ent_coef=0.01,          # Un poco de entropía para ayudar a descubrir cómo esquivar
    device="cpu",           # O "cuda" si el kernel te deja
    tensorboard_log=LOG_DIR
)

# --- CALLBACKS ---
eval_cb = EvalCallback(
    eval_env, 
    best_model_save_path=MODEL_DIR, 
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ, 
    n_eval_episodes=20,     # Evaluamos 20 episodios para tener una métrica sólida
    deterministic=True
)

ckpt_cb = CheckpointCallback(
    save_freq=EVAL_FREQ,
    save_path=os.path.join(MODEL_DIR, "checkpoints"),
    name_prefix="ppo_nuevo_estado"
)

print(f"Iniciando entrenamiento de 0 con {TOTAL_TIMESTEPS} steps...")
print(f"Observación por robot: {train_env.unwrapped.obs_dim_per_robot} valores")

# --- ENTRENAMIENTO ---
t0 = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    callback=CallbackList([eval_cb, ckpt_cb]),
    reset_num_timesteps=True
)

print(f"Entrenamiento finalizado en: {time.time() - t0:.2f}s")

# Guardar resultado final
model.save(os.path.join(MODEL_DIR, "ppo_final_v24_velocidades.zip"))

train_env.close()
eval_env.close()