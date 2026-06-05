import numpy as np
import os
import time  # <--- NUEVO: Necesario para controlar la velocidad del renderizado
import mujoco.viewer  # <--- NUEVO: El visor nativo de MuJoCo
from stable_baselines3 import PPO
from env.satellite_env import SatelliteEnv

# ===============================
# CONFIGURACIÓN
# ===============================
MODEL_PATH = "models/multi_agent/robot_4/ppo_simple_61/checkpoints/ppo_finetune_suave_10380000_steps.zip"

NUM_EPISODES = 50
MAX_STEPS = 2000
DEVICE = "cpu"

SUCCESS_POS_TOL = 0.12      
SUCCESS_SPEED_TOL = 0.08    
COLLISION_DIST = 0.26       
F_MAX_REAL = 0.6            # El límite nominal de tus propulsores en MuJoCo

# ===============================
# CREAR ENTORNO
# ===============================
eval_env = SatelliteEnv(max_steps=MAX_STEPS, n_agents=4)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: No se encuentra el modelo en {MODEL_PATH}")
else:
    model = PPO.load(MODEL_PATH, env=eval_env, device=DEVICE)

    stats = {
        "success": 0,
        "collisions": 0,
        "rewards": [],
        "steps": [],
        "final_dists": [],
        "final_speeds": []
    }

    # --- MÉTRICAS DEL EXPERIMENTO DE ACTUADORES ---
    actuator_stats = {
        "total_env_steps": 0,
        "brake_active_steps": 0,    # Cuántos pasos se usó el freno en total
        "max_brake_force": 0.0,     # El pico más alto registrado de fuerza de frenado
        "max_combined_force": 0.0,  # El pico más alto de (PPO + Freno)
        "saturation_events": 0      # Cuántas veces se superó el límite físico de 0.6 N
    }

    print(f"Evaluando modelo para {eval_env.num_robots} agentes: {MODEL_PATH}...")

    # =========================================================================
    # 🚀 NUEVO: ABRIR EL VISOR PASIVO DE MUJOCO
    # =========================================================================
    # Se le pasa el modelo y los datos que maneja tu SimWrapper/SatelliteEnv
    with mujoco.viewer.launch_passive(eval_env.sim.model, eval_env.sim.data) as viewer:
        
        for ep in range(NUM_EPISODES):
            obs, info = eval_env.reset(seed=2000 + ep)
            
            # Sincronizamos el visor nada más reiniciar el entorno
            viewer.sync()
            print("-"*60)

            done = False
            truncated = False
            ep_reward = 0.0
            step_count = 0
            collision_detected = False
            body_to_geom = {name: f"cube_{i+1}" for i, name in enumerate(eval_env.sim.robot_names)}

            while not (done or truncated):
                # Verificar si el usuario ha cerrado la ventana de MuJoCo manualmente
                if not viewer.is_running():
                    print("Visor cerrado por el usuario. Saliendo...")
                    break

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                ep_reward += reward
                step_count += 1
                
                # =========================================================================
                # 🚀 NUEVO: ACTUALIZAR EL VISOR Y CONTROLAR EL REALISMO TEMPORAL
                # =========================================================================
                viewer.sync()  # Refresca la ventana con los nuevos datos de física
                # Tus pasos internos simulan 5 sub-pasos de dt=0.01 (0.05s por step). 
                # Con sleep(0.02) se verá fluido y rápido, pero cómodo de analizar.
                time.sleep(0.02) 
                
                # --- TELEMETRÍA DE ACTUADORES COMPATIBLE ---
                telemetry = info["actuator_telemetry"]

                for name in eval_env.sim.robot_names:
                    actuator_stats["total_env_steps"] += 1 
                    
                    f_brake_mag = telemetry[name]["max_f_brake"]
                    f_combined_mag = telemetry[name]["max_f_combined"]
                    
                    if f_brake_mag > actuator_stats["max_brake_force"]:
                        actuator_stats["max_brake_force"] = f_brake_mag
                    if f_combined_mag > actuator_stats["max_combined_force"]:
                        actuator_stats["max_combined_force"] = f_combined_mag
                    
                    if f_brake_mag > 0.01:
                        actuator_stats["brake_active_steps"] += 1
                    
                    if f_combined_mag > F_MAX_REAL:
                        actuator_stats["saturation_events"] += 1

                # --- VERIFICACIÓN DE COLISIÓN FÍSICA REAL (MuJoCo) ---
                for i in range(eval_env.sim.data.ncon):
                    contact = eval_env.sim.data.contact[i]
                    if contact.dist < 0:
                        g1 = eval_env.sim.model.geom(contact.geom1).name
                        g2 = eval_env.sim.model.geom(contact.geom2).name
                    
                        for name_i, geom_i in body_to_geom.items():
                            for name_j, geom_j in body_to_geom.items():
                                if name_i != name_j:
                                    if (g1 == geom_i and g2 == geom_j) or (g1 == geom_j and g2 == geom_i):
                                        collision_detected = True
                                        break 
                            if collision_detected: break 
                    if collision_detected: break 

            # Si el usuario cierra el visor en mitad de un bucle, salimos del bucle general
            if not viewer.is_running():
                break

            # --- Procesamiento Fin de Episodio ---
            final_dists = []
            final_speeds = []
            robots_ok = 0
            
            for name in eval_env.sim.robot_names:
                p = eval_env.sim.data.xpos[eval_env.body_id[name]][:2].copy()
                target = eval_env.assigned_target_pos[name]
                d = np.linalg.norm(p - target)
                
                vx = eval_env.sim.data.qvel[eval_env.dadr_x[name]]
                vy = eval_env.sim.data.qvel[eval_env.dadr_y[name]]
                v = np.sqrt(vx**2 + vy**2)
                
                final_dists.append(d)
                final_speeds.append(v)
                
                if d < SUCCESS_POS_TOL and v < SUCCESS_SPEED_TOL:
                    robots_ok += 1

            is_success = (robots_ok == eval_env.num_robots)
            
            if is_success: stats["success"] += 1
            if collision_detected: stats["collisions"] += 1
            
            stats["rewards"].append(ep_reward)
            stats["steps"].append(step_count)
            stats["final_dists"].append(np.mean(final_dists))
            stats["final_speeds"].append(np.mean(final_speeds))
                
            print(f"-> Resumen Ep {ep+1:02d} | {'SUCCESS' if is_success else 'FAIL'} | "
                  f"Robots OK: {robots_ok}/{eval_env.num_robots} | "
                  f"Steps: {step_count} | Coll: {collision_detected} | "
                  f"Reward: {ep_reward:.2f}")
            print("="*60)

    # ===============================
    # RESUMEN FINAL + MÉTRICAS TFG
    # ===============================
    if actuator_stats["total_env_steps"] > 0:
        pct_brake_use = (actuator_stats["brake_active_steps"] / actuator_stats["total_env_steps"]) * 100
    else:
        pct_brake_use = 0.0

    print("\n" + "="*50)
    print(f"      RESULTADOS FINALES ({eval_env.num_robots} AGENTES)")
    print("="*50)
    print(f"Tasa de Éxito:       {stats['success']/NUM_EPISODES*100:>6.1f}% ({stats['success']}/{NUM_EPISODES})")
    print(f"Tasa de Colisión:    {stats['collisions']/NUM_EPISODES*100:>6.1f}% ({stats['collisions']}/{NUM_EPISODES})")
    print(f"Recompensa Media:    {np.mean(stats['rewards']):>8.2f}")
    print(f"Pasos Medios:        {np.mean(stats['steps']):>8.1f}")
    print(f"Distancia Final Med: {np.mean(stats['final_dists']):>8.3f} m")
    print(f"Velocidad Final Med: {np.mean(stats['final_speeds']):>8.3f} m/s")
    
    print("\n" + "="*50)
    print("  📊 INFORME DE SATURACIÓN DE ACTUADORES (ANÁLISIS TFG)")
    print("="*50)
    print(f"Uso del Freno de Emergencia:    {pct_brake_use:.3f}% de la misión.")
    print(f"Fuerza Máxima del Freno:        {actuator_stats['max_brake_force']:.3f} N")
    print(f"Fuerza Máxima Combinada:        {actuator_stats['max_combined_force']:.3f} N (Límite nominal: {F_MAX_REAL} N)")
    print(f"Eventos de Supersaturación:     {actuator_stats['saturation_events']} rebasamientos físicos.")
    print("="*50)

eval_env.close()