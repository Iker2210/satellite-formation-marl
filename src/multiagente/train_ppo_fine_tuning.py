import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from env.satellite_env import SatelliteEnv

# --- CONFIGURACIÓN DE DISPOSITIVO ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forzar CPU para evitar errores de kernel CUDA

# --- RUTAS ---
# Cargamos el checkpoint estrella de 9.45M de pasos
# Update paths before running
CHECKPOINT_PATH = "models/multi_agent/robot_4/ppo_simple_60/checkpoints/ppo_finetune_suave_10350000_steps.zip"

LOG_DIR = "logs/robot_4/ppo_simple_61"
MODEL_DIR = "models/multi_agent/robot_4/ppo_simple_61"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- HIPERPARÁMETROS DE AJUSTE FINO ---
TOTAL_TIMESTEPS = 300_000   # 3M de pasos para asimilar las perturbaciones orbitales
LEARNING_RATE = 3e-5          # Conservador para guiar la política existente
SEED = 42
EVAL_FREQ = 30_000
ENV_MAX_STEPS = 2000


def make_env(seed: int):
    env = SatelliteEnv(max_steps=ENV_MAX_STEPS, verbose=False, seed=seed, n_agents=4)
    env = Monitor(env)
    return env


# --- Crear entornos ---
train_env = make_env(SEED)
eval_env = make_env(SEED + 2000)

# Forzar evaluación al 100% de perturbaciones (peor caso, no diluido por el currículum)
# Accedemos a través de .env porque está envuelto en Monitor
eval_env.env.perturbation_scale = 1.0
eval_env.env.perturbation_increment = 0.0

# --- CARGAR MODELO EXISTENTE ---
if os.path.exists(CHECKPOINT_PATH):
    print(f"Cargando modelo previo: {CHECKPOINT_PATH}")

    # ------------ MODIFICACIÓN FINE-TUNING ULTRA CONSERVADOR ------------
    new_lr_schedule = lambda _: LEARNING_RATE

    custom_objects = {
        "learning_rate": new_lr_schedule,
        "ent_coef": 0.001,           # Casi 0 para evitar que den bandazos aleatorios
        "clip_range": 0.1,           # Cambios más pequeños por update para no romper la navegación
        "n_steps": 4096,
        "tensorboard_log": LOG_DIR,
    }
    # --------------------------------------------------------------------

    model = PPO.load(CHECKPOINT_PATH,
                     env=train_env,
                     device="cpu",
                     custom_objects=custom_objects)
else:
    raise FileNotFoundError(f"No se encontró el checkpoint en {CHECKPOINT_PATH}")

# --- CALLBACKS ---
eval_cb = EvalCallback(eval_env,
                       best_model_save_path=MODEL_DIR,
                       log_path=LOG_DIR,
                       eval_freq=EVAL_FREQ,
                       n_eval_episodes=15,
                       deterministic=True)

ckpt_cb = CheckpointCallback(save_freq=EVAL_FREQ,
                             save_path=os.path.join(MODEL_DIR, "checkpoints"),
                             name_prefix="ppo_finetune_suave")

print("Iniciando Fine-Tuning con perturbaciones orbitales (PPO_50)...")
print(f"Target de pasos en Tensorboard: de 9.45M a 12.45M")
print(f"Currículum perturbaciones: 0.0 → 1.0 en ~200k pasos de RL")
print(f"Evaluación con perturbaciones FIJAS al 100%")

# --- ENTRENAMIENTO ---
t0 = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS,
            callback=CallbackList([eval_cb, ckpt_cb]),
            reset_num_timesteps=False,    # Sigue desde los 9.45M del checkpoint anterior
            tb_log_name="ppo_simple_50")

print(f"Entrenamiento finalizado en: {time.time() - t0:.2f}s")

# Guardar resultado final
model.save(os.path.join(MODEL_DIR, "ppo_final_perturbaciones.zip"))

train_env.close()
eval_env.close()