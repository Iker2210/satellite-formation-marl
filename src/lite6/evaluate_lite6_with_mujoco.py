import os
import numpy as np
import mujoco.viewer
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from lite6_new_env import Lite6Env 

# =========================================================
# CONFIG
# =========================================================
# Update BASE_DIR to your local path before running
BASE_DIR = "/home/iker/TFG_RL_robots/lite6_validation/control"
MODEL_DIR = os.path.join(BASE_DIR, "models")

RUN_ID = 51
RUN_DIR = os.path.join(MODEL_DIR, f"run_{RUN_ID}")
MODEL_PATH = os.path.join(RUN_DIR, "checkpoints/ppo_lite6_run51_950000_steps.zip")
STATS_PATH = os.path.join(RUN_DIR, "vec_normalize.pkl") # Importante para la normalización

NUM_EPISODES = 20
MAX_STEPS = 3000 

# Variables para métricas
success_count = 0
total_steps = 0
distances_log = []

# =========================================================
# ENV + MODEL
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
# VIEWER
# =========================================================
viewer = mujoco.viewer.launch_passive(raw_env.model, raw_env.data)

# =========================================================
# EVALUACIÓN
# =========================================================
print(f"\n📊 Iniciando evaluación de {NUM_EPISODES} episodios...")

for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    step = 0
    ep_success = False
    
    # --- Variable para calcular velocidad ---
    # Obtenemos posición inicial del EE
# --- Variable para calcular velocidad ---
    # Cambiamos 'ee' por 'attachment_site' que es el nombre válido según el error
    # --- Variable para calcular velocidad ---
    site_id = raw_env.model.site('attachment_site').id
    last_ee_pos = raw_env.data.site_xpos[site_id].copy()
    
    # Tiempo real entre cada paso de la IA (segundos)
    f_skip = getattr(raw_env, 'frame_skip', 1)
    dt = raw_env.model.opt.timestep * f_skip 

    print(f"\n--- EPISODIO {ep+1} ---")
    print(f"Target: {raw_env.ee_target}")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # --- Cálculo de Velocidad ---
        current_ee_pos = raw_env.data.site_xpos[site_id].copy()
        dist_moved = np.linalg.norm(current_ee_pos - last_ee_pos)
        velocity = dist_moved / dt  # m/s
        last_ee_pos = current_ee_pos.copy()

        viewer.sync()
        time.sleep(0.005) 

        step += 1
        done = dones[0]
        current_dist = infos[0].get("dist", 1.0)
        
        if current_dist < 0.06:
            ep_success = True

        if step % 100 == 0:
            print(f"Step {step:4} | Dist: {current_dist:.4f}m | Vel: {velocity:.4f} m/s")

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
viewer.close()
# ... (resto del resumen final igual)
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

#####TEST ESTRES PUNTOS CONFLICTIVOS
#import os
#import numpy as np
#import mujoco.viewer
#import time
#from stable_baselines3 import PPO
#from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
#from lite6_new_env import Lite6Env 
#
## =========================================================
## CONFIGURACIÓN
## =========================================================
#BASE_DIR = "/home/iker/TFG_RL_robots/lite6_validation/control"
#MODEL_DIR = os.path.join(BASE_DIR, "models")
#
#RUN_ID = 37 # Tu último run entrenado con J3 abierta
#RUN_DIR = os.path.join(MODEL_DIR, f"run_{RUN_ID}")
#MODEL_PATH = os.path.join(RUN_DIR, "best_model.zip")
#STATS_PATH = os.path.join(RUN_DIR, "vec_normalize.pkl") 
#
## Definimos los puntos críticos en la zona negativa (X, Y, Z)
## He incluido puntos lejanos (-0.6), medios (-0.4) y cercanos (-0.2)
#STRESS_TARGETS = [
#    [-0.6, -0.3, 0.21], # Esquina extrema izquierda
#    [-0.5, -0.3, 0.21], 
#    [-0.4, -0.3, 0.21], 
#    [-0.3, -0.2, 0.21], 
#    [-0.2, -0.3, 0.21], # Muy cerca de la base, lateral izquierdo
#    [-0.1, -0.3, 0.21], # Punto de máxima flexión
#    [-0.6, -0.1, 0.21], # Lejano, casi central
#]
#
#NUM_EPISODES = len(STRESS_TARGETS)
#MAX_STEPS = 2500 
#
## Variables para métricas
#success_count = 0
#total_steps = 0
#distances_log = []
#
## =========================================================
## ENV + MODEL
## =========================================================
#def make_env():
#    return Lite6Env(max_steps=MAX_STEPS)
#
#env = DummyVecEnv([make_env])
#
#if os.path.exists(STATS_PATH):
#    env = VecNormalize.load(STATS_PATH, env)
#    env.training = False
#    env.norm_reward = False 
#    print("✅ Estadísticas de normalización cargadas correctamente.")
#else:
#    print("⚠️ ADVERTENCIA: No se encontró vec_normalize.pkl.")
#
#model = PPO.load(MODEL_PATH, env=env, device="cpu")
#raw_env = env.envs[0].unwrapped
#
## =========================================================
## VIEWER
## =========================================================
#viewer = mujoco.viewer.launch_passive(raw_env.model, raw_env.data)
#
## =========================================================
## EVALUACIÓN DE ESTRÉS
## =========================================================
#print(f"\n🧪 Iniciando TEST DE ESTRÉS en Zona Negativa ({NUM_EPISODES} puntos)...")
#
#for ep in range(NUM_EPISODES):
#    # 1. Reset
#    obs = env.reset()
#    
#    # 2. Forzar target manual en el raw_env
#    target_fijo = np.array(STRESS_TARGETS[ep], dtype=np.float32)
#    raw_env.ee_target = target_fijo
#    raw_env._update_visual_target()
#    
#    # 3. RE-CALCULAR OBSERVACIÓN MANUALMENTE
#    # Esto es vital: actualizamos la observación del wrapper con el nuevo target
#    raw_obs = raw_env._get_obs()
#    obs = env.normalize_obs(raw_obs.reshape(1, -1)) # Forzamos formato (1, 25)
#
#    done = False
#    step = 0
#    ep_success = False
#
#    print(f"\n--- TEST PUNTO {ep+1} ---")
#    print(f"Target Crítico: {target_fijo}")
#
#    while not done:
#        # deterministic=True para evitar ruidos en el test
#        action, _ = model.predict(obs, deterministic=True)
#        
#        # EL TRUCO: Asegurarnos de que action sea un array de (1, 3)
#        # VecEnv siempre espera un batch de acciones
#        if action.ndim == 1:
#            action = action.reshape(1, -1)
#
#        obs, rewards, dones, infos = env.step(action)
#        
#        viewer.sync()
#
#        step += 1
#        done = dones[0]
#        
#        current_dist = infos[0].get("dist", 1.0)
#        
#        if current_dist < 0.06:
#            ep_success = True
#
#        if step % 500 == 0:
#            print(f"Step {step} | Dist: {current_dist:.4f}m")
#
#    # Guardar métricas
#    if ep_success:
#        success_count += 1
#        print(f"✅ RESULTADO: ÉXITO en {step} pasos")
#    else:
#        print(f"❌ RESULTADO: FALLO (Distancia final: {current_dist:.4f}m)")
#    
#    total_steps += step
#    distances_log.append(current_dist)
#    time.sleep(0.5)
#
## =========================================================
## RESUMEN FINAL
## =========================================================
#viewer.close()
#
#success_rate = (success_count / NUM_EPISODES) * 100
#avg_dist = np.mean(distances_log)
#
#print("\n" + "="*40)
#print("       RESUMEN TEST DE ESTRÉS")
#print("="*40)
#print(f"Puntos evaluados:   {NUM_EPISODES}")
#print(f"Tasa de Éxito:      {success_rate:.2f}%")
#print(f"Error medio final:  {avg_dist:.4f} m")
#print(f"Pasos medios/ep:    {total_steps / NUM_EPISODES:.1f}")
#print("="*40)