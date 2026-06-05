import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Importamos tu clase del archivo lite6_env.py
from lite6_new_env import Lite6Env

# =========================================================
# 1. CONFIGURACIÓN DE RUTAS Y VERSIONADO
# =========================================================
# Update BASE_DIR to your local path before running
BASE_DIR = "/home/iker/TFG_RL_robots/lite6_validation/control"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Auto-incremento de la carpeta de ejecución (run_1, run_2...)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    
existing_runs = [d for d in os.listdir(MODEL_DIR) if d.startswith("run_")]
RUN_ID = len(existing_runs) + 1

RUN_DIR = os.path.join(MODEL_DIR, f"run_{RUN_ID}")
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
LOG_DIR = os.path.join(RUN_DIR, "logs")
TB_DIR = os.path.join(RUN_DIR, "tensorboard")

for d in [RUN_DIR, CHECKPOINT_DIR, LOG_DIR, TB_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"\n🚀 Iniciando entrenamiento RUN_{RUN_ID}")

# =========================================================
# 2. CREACIÓN DEL ENTORNO (CON NORMALIZACIÓN)
# =========================================================
def make_env():
    # max_steps=300 es un buen compromiso entre exploración y tiempo
    env = Lite6Env(max_steps=2500)
    env = Monitor(env)
    return env

# Vectorizamos el entorno (necesario para VecNormalize)
env = DummyVecEnv([make_env])

# VecNormalize: Escala las observaciones y recompensas automáticamente. 
# Es VITAL en robótica para que el entrenamiento no diverja.
env = VecNormalize(
    env, 
    norm_obs=True, 
    norm_reward=True, 
    clip_obs=10.0,
    gamma=0.99
)

# Entorno separado para evaluación (no entrena, solo mide rendimiento)
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
# Sincroniza las estadísticas iniciales
eval_env.obs_rms = env.obs_rms
# =========================================================
# 3. CONFIGURACIÓN DE CALLBACKS
# =========================================================
# Guarda el modelo cada 50,000 pasos
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=CHECKPOINT_DIR,
    name_prefix=f"ppo_lite6_run{RUN_ID}"
)

# Evalúa el robot cada 10,000 pasos y guarda el "mejor modelo" hasta la fecha
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=RUN_DIR,
    log_path=LOG_DIR,
    eval_freq=25000,
    n_eval_episodes=10,
    deterministic=True
)

callbacks = CallbackList([checkpoint_callback, eval_callback])

# =========================================================
# 4. DEFINICIÓN DEL MODELO PPO
# =========================================================
# Ajustamos hiperparámetros para control continuo con inercia
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",               # Cambia a "cuda" si tienes GPU Nvidia
    learning_rate=1e-4,         # Tasa estándar para robótica
    n_steps=4096,               # Horizonte de recolección de datos (más largo = más estable)
    batch_size=128, #64,              # Tamaño del mini-batch para optimizar
    n_epochs=10,                # Cuántas veces procesa cada batch de datos
    gamma=0.99,                 # Factor de descuento (importancia de premios futuros)
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,             # Un poco de entropía para fomentar la exploración
    policy_kwargs=dict(net_arch=[256, 256]), # Red un poco más profunda
    tensorboard_log=TB_DIR
)

# =========================================================
# 5. ENTRENAMIENTO
# =========================================================
TIMESTEPS = 500_000 # El robot necesita tiempo para entender el deslizamiento

try:
    print(f"Entrenando por {TIMESTEPS} pasos...")
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callbacks,
        tb_log_name="PPO_run"
    )
except KeyboardInterrupt:
    print("\n⚠️ Entrenamiento interrumpido por el usuario.")

# =========================================================
# 6. GUARDADO FINAL
# =========================================================
# ¡IMPORTANTE! Debes guardar el modelo Y las estadísticas de normalización.
# Sin el archivo .pkl, el modelo cargado no sabrá qué escala usar.
final_model_path = os.path.join(RUN_DIR, "final_model")
model.save(final_model_path)

stats_path = os.path.join(RUN_DIR, "vec_normalize.pkl")
env.save(stats_path)

print(f"\n✅ Entrenamiento finalizado.")
print(f"Modelo: {final_model_path}")
print(f"Estadísticas de normalización: {stats_path}")
print(f"Visualiza el progreso con: tensorboard --logdir={TB_DIR}")