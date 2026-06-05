import os
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from lite6_new_env import Lite6Env

# =========================================================
# CONFIGURACIÓN
# =========================================================
# Update BASE_DIR to your local path before running
BASE_DIR = "/home/iker/TFG_RL_robots/lite6_validation/control"
MODEL_DIR = os.path.join(BASE_DIR, "models")

RUN_ID = 51
RUN_DIR = os.path.join(MODEL_DIR, f"run_{RUN_ID}")
MODEL_PATH = os.path.join(RUN_DIR, "checkpoints/ppo_lite6_run51_950000_steps.zip")
STATS_PATH = os.path.join(RUN_DIR, "vec_normalize.pkl") # Importante para la normalización

NUM_EPISODES = 20
MAX_STEPS = 3000 # Sincronizado con el entrenamiento

# Variables para métricas
success_count = 0
total_steps = 0
distances_log = []

# =========================================================
# ENV + MODEL (CON NORMALIZACIÓN)
# =========================================================
def make_env():
    return Lite6Env(max_steps=MAX_STEPS)

env = DummyVecEnv([make_env])

if os.path.exists(STATS_PATH):
    env = VecNormalize.load(STATS_PATH, env)
    env.training = False
    env.norm_reward = False 
    print("✅ Estadísticas de normalización cargadas correctamente.")
else:
    print("⚠️ ADVERTENCIA: No se encontró vec_normalize.pkl.")

model = PPO.load(MODEL_PATH, env=env, device="cpu")
raw_env = env.envs[0].unwrapped

# =========================================================
# EVALUACIÓN
# =========================================================
print(f"\n📊 Iniciando evaluación de {NUM_EPISODES} episodios...")

for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    step = 0
    ep_success = False

    print(f"\n--- EPISODIO {ep+1} ---")
    print(f"Target: {raw_env.ee_target}")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
    
        step += 1
        done = dones[0]
        
        # Extraer info
        current_dist = infos[0].get("dist", 1.0)
        
        # Verificamos éxito: si en algún paso del episodio se alcanzó el umbral
        if current_dist < 0.06:
            ep_success = True

        if step % 100 == 0:
            print(f"Step {step} | Dist: {current_dist:.4f}m")

    # Guardar métricas
    if ep_success:
        success_count += 1
        print("✅ RESULTADO: ÉXITO")
    else:
        print(f"❌ RESULTADO: FALLO (Distancia final: {current_dist:.4f}m)")
    
    total_steps += step
    distances_log.append(current_dist)
    time.sleep(0.5)

# =========================================================
# RESUMEN FINAL
# =========================================================

success_rate = (success_count / NUM_EPISODES) * 100
avg_dist = np.mean(distances_log)

print("\n" + "="*30)
print("       RESUMEN FINAL")
print("="*30)
print(f"Episodios evaluados: {NUM_EPISODES}")
print(f"Tasa de Éxito:      {success_rate:.2f}%")
print(f"Distancia Media:    {avg_dist:.4f} m")
print(f"Pasos medios/ep:    {total_steps / NUM_EPISODES:.1f}")
print("="*30)