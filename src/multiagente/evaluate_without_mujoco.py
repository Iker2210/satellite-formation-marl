"""
Script de evaluación de la calidad geométrica de la formación.

Mide tres métricas estructurales sobre N_EPISODIOS:
  1. Cohesión: distancia media de los agentes al centroide de la formación,
     promediada sobre los últimos VENTANA_PASOS pasos del episodio (formación
     ya consolidada).
  2. Separación mínima: distancia mínima entre dos agentes cualesquiera
     observada a lo largo de todo el episodio (peor caso de aproximación).
  3. Error angular medio: desviación media de la separación angular observada
     respecto a la separación ideal 2*pi/N, promediada sobre los últimos
     VENTANA_PASOS pasos.

Las métricas se reportan agregadas sobre el conjunto de episodios EXITOSOS
(la separación mínima también se reporta agregada sobre todos los episodios
para obtener el peor caso global, no solo el de los episodios donde la
política funcionó).
"""

import os
import numpy as np
from stable_baselines3 import PPO
from env.satellite_env import SatelliteEnv

# ===============================
# CONFIGURACIÓN
# ===============================
MODEL_PATH = "models/multi_agent/robot_4/ppo_simple_61/checkpoints/ppo_finetune_suave_10380000_steps.zip"
NUM_AGENTS = 4
NUM_EPISODES = 1000           # 1000 para PPO_final; reduce a 100 si vas con prisa
MAX_STEPS = 2000
DEVICE = "cpu"

# Tolerancias de éxito
SUCCESS_POS_TOL = 0.12
SUCCESS_SPEED_TOL = 0.08

# Ventana final (en pasos) para promediar cohesión y error angular.
# Mide la calidad de la formación YA ALCANZADA, no del transitorio.
VENTANA_PASOS = 100

# ===============================
# CREAR ENTORNO Y MODELO
# ===============================
eval_env = SatelliteEnv(max_steps=MAX_STEPS, n_agents=NUM_AGENTS)
assert os.path.exists(MODEL_PATH), f"No se encuentra el modelo: {MODEL_PATH}"
model = PPO.load(MODEL_PATH, env=eval_env, device=DEVICE)

# Separación angular ideal entre agentes (2*pi/N)
sep_ideal = 2 * np.pi / NUM_AGENTS

# ===============================
# AGREGADOS A LO LARGO DE TODOS LOS EPISODIOS
# ===============================
cohesion_por_ep = []          # cohesión final (ventana) por episodio exitoso
error_ang_por_ep = []         # error angular medio (ventana) por episodio exitoso
sep_min_global = []           # separación mínima de todo el episodio (todos)
sep_min_exitosos = []         # ídem, solo episodios exitosos

n_exitos = 0
n_colisiones = 0

print(f"\nEvaluando {NUM_EPISODES} episodios con {NUM_AGENTS} agentes...")
print(f"Modelo: {MODEL_PATH}\n")

for ep in range(NUM_EPISODES):
    obs, info = eval_env.reset(seed=2000 + ep)

    # --- Trackers del episodio ---
    positions_history = []        # (T, N, 2) — para cohesión y error angular
    sep_min_episodio = np.inf     # peor aproximación observada en este episodio

    done = False
    truncated = False
    collision_detected = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)

        # Posiciones actuales de los N agentes (plano XY)
        positions = np.array([
            eval_env.sim.data.xpos[eval_env.body_id[name]][:2].copy()
            for name in eval_env.sim.robot_names
        ])
        positions_history.append(positions)

        # Separación mínima en este paso (entre cualquier par)
        for i in range(NUM_AGENTS):
            for j in range(i + 1, NUM_AGENTS):
                d_ij = np.linalg.norm(positions[i] - positions[j])
                if d_ij < sep_min_episodio:
                    sep_min_episodio = d_ij

        # Verificación de colisión física real (igual que en el script principal)
        if not collision_detected:
            body_to_geom = {name: f"cube_{i+1}"
                            for i, name in enumerate(eval_env.sim.robot_names)}
            for i in range(eval_env.sim.data.ncon):
                contact = eval_env.sim.data.contact[i]
                if contact.dist < 0:
                    g1 = eval_env.sim.model.geom(contact.geom1).name
                    g2 = eval_env.sim.model.geom(contact.geom2).name
                    for name_i, geom_i in body_to_geom.items():
                        for name_j, geom_j in body_to_geom.items():
                            if name_i != name_j:
                                if (g1 == geom_i and g2 == geom_j) or \
                                   (g1 == geom_j and g2 == geom_i):
                                    collision_detected = True
                                    break
                        if collision_detected:
                            break
                if collision_detected:
                    break

    # --- Cierre de episodio: determinar si fue éxito ---
    robots_ok = 0
    for name in eval_env.sim.robot_names:
        p = eval_env.sim.data.xpos[eval_env.body_id[name]][:2].copy()
        target = eval_env.assigned_target_pos[name]
        d = np.linalg.norm(p - target)
        vx = eval_env.sim.data.qvel[eval_env.dadr_x[name]]
        vy = eval_env.sim.data.qvel[eval_env.dadr_y[name]]
        v = np.sqrt(vx**2 + vy**2)
        if d < SUCCESS_POS_TOL and v < SUCCESS_SPEED_TOL:
            robots_ok += 1
    is_success = (robots_ok == NUM_AGENTS) and not collision_detected

    if is_success:
        n_exitos += 1
    if collision_detected:
        n_colisiones += 1

    # Separación mínima: la registramos siempre
    sep_min_global.append(sep_min_episodio)
    if is_success:
        sep_min_exitosos.append(sep_min_episodio)

    # Cohesión y error angular: solo tienen sentido si el episodio fue exitoso
    # (en fallos, los agentes no están sobre el anillo y el dato es ruido)
    if is_success and len(positions_history) >= VENTANA_PASOS:
        ultimos = np.array(positions_history[-VENTANA_PASOS:])  # (T, N, 2)

        # --- Cohesión: distancia media al centroide en cada paso, promediada ---
        centroide = ultimos.mean(axis=1, keepdims=True)         # (T, 1, 2)
        d_al_centroide = np.linalg.norm(ultimos - centroide, axis=2)  # (T, N)
        cohesion_ep = d_al_centroide.mean()
        cohesion_por_ep.append(cohesion_ep)

        # --- Error angular medio respecto a la separación ideal ---
        # Usamos el target global como centro del anillo (donde está centrado)
        ring_center = eval_env.target_pos_global

        errores_ang_paso = []
        for pos_paso in ultimos:
            # Ángulo de cada agente respecto al centro del anillo
            angulos = np.arctan2(pos_paso[:, 1] - ring_center[1],
                                 pos_paso[:, 0] - ring_center[0])
            angulos = np.sort(angulos)
            # Separaciones angulares consecutivas (circular)
            deltas = np.diff(np.concatenate([angulos, [angulos[0] + 2*np.pi]]))
            error_paso = np.mean(np.abs(deltas - sep_ideal))
            errores_ang_paso.append(np.degrees(error_paso))
        error_ang_ep = np.mean(errores_ang_paso)
        error_ang_por_ep.append(error_ang_ep)

    if (ep + 1) % 50 == 0:
        print(f"  Episodio {ep+1:4d}/{NUM_EPISODES} | "
              f"Éxitos: {n_exitos} | Colisiones: {n_colisiones}")

# ===============================
# RESUMEN
# ===============================
print("\n" + "=" * 60)
print(f"   CALIDAD GEOMÉTRICA DE LA FORMACIÓN ({NUM_AGENTS} AGENTES)")
print("=" * 60)
print(f"Episodios evaluados:            {NUM_EPISODES}")
print(f"Tasa de éxito:                  {n_exitos/NUM_EPISODES*100:.1f}% "
      f"({n_exitos}/{NUM_EPISODES})")
print(f"Tasa de colisión:               {n_colisiones/NUM_EPISODES*100:.1f}% "
      f"({n_colisiones}/{NUM_EPISODES})")
print("-" * 60)
print(f"Métricas sobre episodios EXITOSOS ({n_exitos} ep.):")
print(f"  Cohesión media (m):           {np.mean(cohesion_por_ep):.3f} "
      f"± {np.std(cohesion_por_ep):.3f}")
print(f"  Error angular medio (°):      {np.mean(error_ang_por_ep):.2f} "
      f"± {np.std(error_ang_por_ep):.2f}")
print(f"  Separación mín. (m):          {np.mean(sep_min_exitosos):.3f} "
      f"(peor caso: {np.min(sep_min_exitosos):.3f})")
print("-" * 60)
print(f"Separación mínima global (m):   "
      f"{np.mean(sep_min_global):.3f} "
      f"(peor caso global: {np.min(sep_min_global):.3f})")
print("=" * 60)
print("\nValores teóricos de referencia para el anillo (r = 0.45 m):")
print(f"  Cohesión ideal:               0.450 m (radio del anillo)")
print(f"  Separación mínima ideal:      "
      f"{2 * 0.45 * np.sin(np.pi / NUM_AGENTS):.3f} m "
      f"(2·r·sin(π/{NUM_AGENTS}))")
print(f"  Error angular ideal:          0.00°")
print("=" * 60)

eval_env.close()