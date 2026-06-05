import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import time
from lite6_new_env import Lite6Env

# Asumiendo que tu clase Lite6Env ya está definida arriba
# o importada desde tu archivo de entorno

def run_diagnostic_tests(env_class):
    print("--- INICIANDO DIAGNÓSTICO DEL ENTORNO ---")
    
    # 1. TEST DE DIMENSIONES (CRÍTICO)
    env = env_class()
    obs, _ = env.reset()
    
    actual_obs_len = len(obs)
    expected_obs_len = env.observation_space.shape[0]
    
    print(f"[TEST 1] Dimensiones de observación:")
    print(f"  - Declaradas en space: {expected_obs_len}")
    print(f"  - Recibidas en reset: {actual_obs_len}")
    
    if actual_obs_len != expected_obs_len:
        print(f"  ❌ ERROR: El vector de observación no coincide con el espacio declarado.")
    else:
        print(f"  ✅ OK: Dimensiones coincidentes.")

    # 2. TEST DE DINÁMICA (CONSERVACIÓN DE MOMENTO)
    print(f"\n[TEST 2] Verificación de física (Base Flotante):")
    env.reset()
    initial_base_pos = env.data.qpos[0:2].copy()
    
    # Forzamos una acción brusca para ver si la base reacciona
    # Movemos J2, J3 y J5 con fuerza
    action = np.array([1.0, 1.0, 1.0], dtype=np.float32) 
    obs, _, _, _, _ = env.step(action)
    new_base_pos = env.data.qpos[0:2].copy()
    
    base_displacement = np.linalg.norm(new_base_pos - initial_base_pos)
    print(f"  - Desplazamiento de la base tras acción: {base_displacement:.4f} m")
    
    if base_displacement < 1e-5:
        print("  ⚠️ ADVERTENCIA: La base apenas se mueve. Revisa si el 'freejoint' está bien configurado o si la fricción es muy alta.")
    else:
        print("  ✅ OK: La base reacciona al movimiento del brazo (Acoplamiento cinemático).")

    # 3. TEST DE VISUALIZACIÓN Y LOOP MANUAL
    print(f"\n[TEST 3] Abriendo visor de MuJoCo (Cierra la ventana para terminar)...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for episode in range(3):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            step_count = 0
            
            print(f"  - Episodio {episode + 1} en marcha...")
            
            while not (terminated or truncated):
                # Acción aleatoria pero suave para testear límites
                action = env.action_space.sample() * 0.5 
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Sincronizar visor
                viewer.sync()
                time.sleep(0.01) # Para que sea humano-perceptible
                
                step_count += 1
            
            print(f"    Acabado en paso {step_count} | Error final: {info['dist']:.4f}")

    print("\n--- DIAGNÓSTICO FINALIZADO ---")

if __name__ == "__main__":
    # Ejecutar el test
    run_diagnostic_tests(Lite6Env)