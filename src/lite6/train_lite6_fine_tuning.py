import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Importamos tu clase del archivo lite6_new_env.py
from lite6_new_env import Lite6Env

# =========================================================
# 1. CONFIGURACIÓN DE RUTAS Y VERSIONADO
# =========================================================
# Update BASE_DIR to your local path before running
BASE_DIR = "/home/iker/TFG_RL_robots/lite6_validation/control"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Configuración del modelo previo a cargar
PREV_RUN_ID = 50
PREV_RUN_DIR = os.path.join(MODEL_DIR, f"run_{PREV_RUN_ID}")
PREV_MODEL_PATH = os.path.join(PREV_RUN_DIR, "checkpoints/ppo_lite6_run50_850000_steps.zip")
PREV_STATS_PATH = os.path.join(PREV_RUN_DIR, "vec_normalize.pkl")

# Auto-incremento para la nueva ejecución (Run 33)
existing_runs = [d for d in os.listdir(MODEL_DIR) if d.startswith("run_")]
RUN_ID = len(existing_runs) + 1

RUN_DIR = os.path.join(MODEL_DIR, f"run_{RUN_ID}")
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
LOG_DIR = os.path.join(RUN_DIR, "logs")
TB_DIR = os.path.join(RUN_DIR, "tensorboard")

for d in [RUN_DIR, CHECKPOINT_DIR, LOG_DIR, TB_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"\n🚀 Continuando entrenamiento: RUN_{PREV_RUN_ID} -> RUN_{RUN_ID}")

# =========================================================
# 2. CREACIÓN DEL ENTORNO
# =========================================================
def make_env():
    # El entorno ya tiene el offset de 0.2 y el umbral de 0.05
    env = Lite6Env(max_steps=3000)
    env = Monitor(env)
    return env

# Entorno de entrenamiento
env = DummyVecEnv([make_env])

# CARGAMOS NORMALIZACIÓN PREVIA (Crucial para no romper el aprendizaje)
if os.path.exists(PREV_STATS_PATH):
    print(f"📊 Cargando estadísticas de normalización desde: {PREV_STATS_PATH}")
    env = VecNormalize.load(PREV_STATS_PATH, env)
    env.training = True
    env.norm_reward = True
else:
    print("⚠️ No se encontraron estadísticas previas. Iniciando normalización desde cero.")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

# Entorno de evaluación
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
eval_env.obs_rms = env.obs_rms # Sincronizamos escalas

# =========================================================
# 3. CALLBACKS
# =========================================================
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=CHECKPOINT_DIR,
    name_prefix=f"ppo_lite6_run{RUN_ID}"
)

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
# 4. CARGA DEL MODELO PPO
# =========================================================
if os.path.exists(PREV_MODEL_PATH):
    print(f"🧠 Cargando pesos del modelo: {PREV_MODEL_PATH}")
    model = PPO.load(
        PREV_MODEL_PATH,
        env=env,
        device="cpu",
        custom_objects={
            "learning_rate": 5e-5, # Bajamos un poco la LR para fine-tuning
        }
    )
else:
    print("⚠️ No se encontró el modelo previo. Creando modelo nuevo.")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-5,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=TB_DIR
    )

# =========================================================
# 5. ENTRENAMIENTO (1 MILLÓN DE STEPS)
# =========================================================
TIMESTEPS = 200_000

try:
    print(f"Iniciando aprendizaje por {TIMESTEPS} pasos...")
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callbacks,
        tb_log_name=f"PPO_run{RUN_ID}",
        reset_num_timesteps=False # Importante: mantiene el contador global de pasos
    )
except KeyboardInterrupt:
    print("\n⚠️ Entrenamiento interrumpido.")

# =========================================================
# 6. GUARDADO FINAL
# =========================================================
final_model_path = os.path.join(RUN_DIR, "final_model")
model.save(final_model_path)

stats_path = os.path.join(RUN_DIR, "vec_normalize.pkl")
env.save(stats_path)

print(f"\n✅ Proceso completado.")
print(f"Nuevo modelo: {final_model_path}")
print(f"Nuevas estadísticas: {stats_path}")