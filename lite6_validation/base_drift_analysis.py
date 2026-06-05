import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from lite6_new_env import Lite6Env

# ============ CONFIGURATION — update before running ============
MODEL_PATH = "models/run_51/best_model.zip"      # Update to your model path
STATS_PATH = "models/run_51/vec_normalize.pkl"   # Update to your stats path

def make_env(): return Lite6Env()
venv = DummyVecEnv([make_env])
venv = VecNormalize.load(STATS_PATH, venv)
venv.training = False
venv.norm_reward = False
model = PPO.load(MODEL_PATH, device='cpu')
env = venv.envs[0]

targets = [[-0.6, 0.25, 0.21], [-0.35, 0.25, 0.21],
           [-0.3, 0.1, 0.21], [-0.5, 0.3, 0.21]]

for target in targets:
    obs = venv.reset()
    env.ee_target = np.array(target, dtype=np.float32)

    base_inicial = env.data.qpos[0:3].copy()
    quat_inicial = env.data.qpos[3:7].copy()
    historial_pos = [base_inicial.copy()]
    historial_quat = [quat_inicial.copy()]

    for step in range(2500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        historial_pos.append(env.data.qpos[0:3].copy())
        historial_quat.append(env.data.qpos[3:7].copy())
        if info[0]['dist'] < 0.06:
            break

    historial_pos = np.array(historial_pos)
    desplaz_neto = (historial_pos[-1] - base_inicial) * 1000  # mm
    recorrido = (historial_pos.max(axis=0) - historial_pos.min(axis=0)) * 1000  # mm

    # Cambio de orientación
    q_ini = historial_quat[0]
    q_fin = historial_quat[-1]
    dot = abs(np.dot(q_ini, q_fin))
    dot = min(1.0, dot)
    cambio_ang_grados = np.degrees(2 * np.arccos(dot))

    print(f"\nTarget {target}:")
    print(f"  Desplazamiento NETO de la base:  {desplaz_neto} mm")
    print(f"  Recorrido MÁXIMO de la base:     {recorrido} mm")
    print(f"  Cambio de orientación de base:   {cambio_ang_grados:.2f}°")