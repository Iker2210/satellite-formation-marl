import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class Lite6Env(gym.Env):
    def __init__(self, max_steps=2500): # Aumentamos steps porque ahora es más lento
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0

        self.model = mujoco.MjModel.from_xml_path("lite6_airbearings.xml")
        self.data = mujoco.MjData(self.model)

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.quat_ref = np.array([0.7071068, -0.7071068, 0, 0])
        # Posición inicial: J1=90º, J2=0.2, J3=1.5(90º), el resto 0
        # Cambiamos J3 de 3.14 a 1.57 (90 grados, posición segura y lejos de límites)
        self.init_joints = np.array([0, 0, 1.6125, 0.0, 0.0, 0.0], dtype=np.float64)
        #self.init_joints = np.array([1.5708, 0.0, 3.1415, 0.0, 0.0, 0.0], dtype=np.float64)
        self.last_action = np.zeros(3) # Para medir brusquedad
        # Acciones: J2, J3, J5
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self.prev_dist = 0.0

        # OBSERVACIÓN (25 elementos):
        # 1. Inclinación Base (Quat): 4
        # 2. Centro de Masas (CoM) relativo a base: 3
        # 3. Vel. Base (Lin + Ang): 6
        # 4. Brazo (Pos/Vel J2, J3, J5): 6
        # 5. Error Relativo EF-Target: 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

        self.ee_target = np.array([-0.4, 0.1, 0.21], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # 1. Estado inicial físico (Mover esto arriba para que _get_ee_pos() lea bien la pose inicial)
        self.data.qpos[0:3] = [0, 0, 0.22]  
        self.data.qpos[3:7] = [0.7071068, -0.7071068, 0, 0] 
        self.data.qpos[7:13] = self.init_joints
        self.data.ctrl[0:6] = self.init_joints
        mujoco.mj_forward(self.model, self.data)

        # 2. SELECCIÓN ALEATORIA DE PUNTOS DE PRUEBA (VALIDACIÓN FÍSICA)
        puntos_prueba = [
            [-0.6, 0.25, 0.21],
            [-0.35, 0.25, 0.21],
            [-0.3, 0.1, 0.21],
            [-0.5, 0.3, 0.21]
        ]
        
        # Escogemos un índice aleatorio de la lista
        idx = self.np_random.integers(0, len(puntos_prueba))
        self.ee_target = np.array(puntos_prueba[idx], dtype=np.float32)

        # 3. Inicializamos prev_dist RELATIVA
        ee_rel = self._get_ee_pos_rel_to_base()
        target_rel = self._get_target_rel_to_base()
        self.prev_dist = np.linalg.norm(ee_rel - target_rel)

        # 4. Estabilización: damos tiempo a que los airbearings se asienten en el suelo
        for _ in range(300):
            mujoco.mj_step(self.model, self.data)

        # Resetear velocidades para empezar limpio (importante: tras el asentamiento
        # puede quedar deriva residual que falsea el inicio del episodio)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        # Recalcular prev_dist después del asentamiento (la base ha bajado ~15mm)
        ee_rel = self._get_ee_pos_rel_to_base()
        target_rel = self._get_target_rel_to_base()
        self.prev_dist = np.linalg.norm(ee_rel - target_rel)
        
        self._update_visual_target()

        return self._get_obs(), {}

    def step(self, action):
        action = action.astype(np.float64) 
        factor_paso = 0.002 # Mantén este, es perfecto para 2500 pasos
        
        # 1. Penalizar cambios bruscos de acción (Smoothness)
        jerk_penalty = np.linalg.norm(action - self.last_action)
        self.last_action = action.copy()

        # 2. Control (target_q[1, 2, 4] corresponden a J2, J3, J5)
        # Usamos self.data.ctrl en lugar de current_q para que el comando sea estable
        new_ctrl = self.data.ctrl[0:6].copy()
        new_ctrl[1] = np.clip(new_ctrl[1] + action[0] * factor_paso, -2.6, 2.6)
        new_ctrl[2] = np.clip(new_ctrl[2] + action[1] * factor_paso, 0.5, 2.2) 
        new_ctrl[4] = np.clip(new_ctrl[4] + action[2] * factor_paso, -2.5, 2.5)
        self.data.ctrl[0:6] = new_ctrl

        for _ in range(15): 
            mujoco.mj_step(self.model, self.data)

        # 3. RECOMPENSA REDISEÑADA
        # Calculamos distancia relativa
        # --- BLOQUE DE RECOMPENSA CORREGIDO ---
        # 1. Usamos SIEMPRE las posiciones RELATIVAS para que la deriva no engañe al brazo
        ee_rel = self._get_ee_pos_rel_to_base()
        target_rel = self._get_target_rel_to_base()
        dist = np.linalg.norm(ee_rel - target_rel) # Esta es la distancia que importa
        
        # 2. Recompensa por estar vivo
        reward = 0.01

        # 3. Recompensa por PROGRESO (Gradiente)
        diff_dist = self.prev_dist - dist
        reward += diff_dist * 200.0 
        self.prev_dist = dist # Guardamos la distancia relativa para el siguiente paso

        torques = self.data.actuator_force.copy()
        reward -= 0.001 * np.sum(np.square(torques))

        # 4. Bonus por cercanía (Exponencial) y penalización por distancia
        reward += np.exp(-5.0 * dist)
        reward -= 0.01 * dist 
        
        # 5. Penalizaciones físicas (Tilt y Vibración)
        # Tilt como desviación angular respecto al quat de referencia
        q_actual = self.data.qpos[3:7]
        # Producto interno entre quats (siempre tomamos valor absoluto por la doble cobertura)
        dot = abs(np.dot(q_actual, self.quat_ref))
        dot = min(1.0, dot)
        tilt = 2 * np.arccos(dot)  # ángulo entre los dos quats, en radianes

        reward -= 5.0 * tilt
        reward -= 0.5 * jerk_penalty

        # Se aumenta el peso del bonus exponencial cuando esté muy cerca
        if dist < 0.1:
            reward += np.exp(-15.0 * dist) * 5.0 # Bonus extra en la zona de aproximación final

        if dist < 0.10: 
            reward -= 0.5 * jerk_penalty
            
            # Necesitas definir compass_local aquí para que el step la reconozca:
            ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
            error_local = ee_mat.T @ (self.ee_target - self._get_ee_pos())
            compass_local = error_local / (dist + 1e-6)
            
            reward += 0.5 * compass_local[2] # Premio por alineación Z
            # Si el robot se aleja (como en tu log), castigamos fuerte
            if diff_dist < 0: 
                # Castigo dinámico: cuanto más rápido se aleje, peor.
                reward += diff_dist * 400.0 # Multiplicador alto para "frenar" la mala dirección

        # 4. TERMINACIÓN
        terminated = False
        base_height = self.data.qpos[2]
        
        # Si inclina más de ~8 grados (0.15 rad), fin del juego
        if base_height < 0.12 or base_height > 0.25 or tilt > 0.20:
            reward = -200.0 
            terminated = True
        
        if dist < 0.06: # Cambiado de 0.03 a 0.05
            reward += 500.0
            terminated = True
            #print("¡SUCCESS! Target alcanzado.")
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        self._update_visual_target()

        return self._get_obs(), float(reward), terminated, truncated, {"dist": dist}

    def _get_obs(self):
        # 1. Mantén lo que funciona
        base_quat = self.data.qpos[3:7].copy()
        com_rel = (self.data.subtree_com[0] - self.data.qpos[0:3]).copy()
        base_vel = self.data.qvel[0:6].copy()

        # 2. Articulaciones (Asegúrate de que J5 esté bien presente)
        q = self.data.qpos[7:13]
        v = self.data.qvel[6:12]
        arm_obs = np.array([q[1], q[2], q[4], v[1], v[2], v[4]])

        # 3. LA CLAVE: Error en coordenadas del sitio "ee_site"
        # Esto le dice al robot si el target está arriba/abajo/izq/der de su mano
        target_pos = self.ee_target
        ee_pos = self._get_ee_pos()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3) # Rotación de la mano
        
        error_global = target_pos - ee_pos
        # Proyectamos el error en el sistema local de la mano
        error_local = ee_mat.T @ error_global 
        
        dist = np.linalg.norm(error_local)
        compass_local = error_local / (dist + 1e-6)

        return np.concatenate([base_quat, com_rel, base_vel, arm_obs, error_local, compass_local]).astype(np.float32)
        
    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()
    
    def _get_ee_pos_rel_to_base(self):
        # Posición del Efector Final en el mundo
        ee_world = self.data.site_xpos[self.ee_site_id].copy()
        # Posición de la base del robot en el mundo
        base_world = self.data.qpos[0:3].copy()
        # Retornamos la posición del brazo respecto a su propia base
        return ee_world - base_world

    def _get_target_rel_to_base(self):
        # El target es un punto fijo en el mundo, pero el robot se mueve
        base_world = self.data.qpos[0:3].copy()
        return self.ee_target - base_world
    
    # Crea esta función dentro de tu clase Lite6Env
    def _update_visual_target(self):
        # Usamos mocapid en lugar del ID de cuerpo general
        mocap_id = self.model.body("visual_target").mocapid[0]
        
        # Si mocapid es -1, significa que el cuerpo no es mocap (error en el XML)
        if mocap_id != -1:
            self.data.mocap_pos[mocap_id] = self.ee_target