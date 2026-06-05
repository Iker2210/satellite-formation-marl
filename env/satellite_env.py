import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from .sim_wrapper import SimWrapper

# ---- IMPORTACIONES DE PERTURBACIONES ----
from physics import OrbitalPerturbations


class SatelliteEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=2000, verbose=False, seed=None, n_agents=4, dynamic_target=False, use_emergency_brake=True):
        super().__init__()

        self._default_max_steps = int(max_steps)
        self.max_steps = self._default_max_steps
        self.verbose = bool(verbose)
        self.current_step = 0
        self.seed = seed

        # ---- MuJoCo ----
        model = mujoco.MjModel.from_xml_path("models/robot_cube.xml")
        data = mujoco.MjData(model)
        self.sim = SimWrapper(model, data)
        self.model = self.sim.model
        self.data = self.sim.data

        # ---- Robots ----
        self.num_robots = n_agents
        self.sim.robot_names = [f"cube_body_{i+1}" for i in range(self.num_robots)]

        # ---- Diccionarios joint ----
        self.body_id = {}
        self.mass = {}
        self.qadr_x = {}
        self.qadr_y = {}
        self.qadr_yaw = {}
        self.dadr_x = {}
        self.dadr_y = {}
        self.dadr_yaw = {}
        self.base_x = {}
        self.base_y = {}

        for i, name in enumerate(self.sim.robot_names):
            self.body_id[name] = self.model.body(name).id
            self.mass[name] = self.sim.model.body_mass[self.body_id[name]]

            jid_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"move_x_{i+1}")
            jid_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"move_y_{i+1}")
            jid_yaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"yaw_{i+1}")

            self.qadr_x[name] = self.model.jnt_qposadr[jid_x]
            self.qadr_y[name] = self.model.jnt_qposadr[jid_y]
            self.qadr_yaw[name] = self.model.jnt_qposadr[jid_yaw]
            self.dadr_x[name] = self.model.jnt_dofadr[jid_x]
            self.dadr_y[name] = self.model.jnt_dofadr[jid_y]
            self.dadr_yaw[name] = self.model.jnt_dofadr[jid_yaw]

            self.base_x[name] = float(self.model.body_pos[self.body_id[name]][0])
            self.base_y[name] = float(self.model.body_pos[self.body_id[name]][1])

        # ---- Área de trabajo ----
        table_x_half = 2.5
        table_y_half = 1.5
        cube_half = 0.15
        safety = 0.10

        self.x_min = -table_x_half + cube_half + safety
        self.x_max = table_x_half - cube_half - safety
        self.y_min = -table_y_half + cube_half + safety
        self.y_max = table_y_half - cube_half - safety
        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min

        # ---- Normalización ----
        self.v_max = 0.5

        # ---- PD interno ----
        self.v_cmd_max = 0.30
        self.kp_v = 0.5
        self.kd_v = 0.1
        self.kd_damping = 1.2
        self.F_max = 0.6
        self.dt = 0.01
        
        # Flag para desactivar el escudo de proximidad físico (estudio ablativo)
        self.use_emergency_brake = bool(use_emergency_brake)

        self.dvx = {}
        self.dvy = {}
        self.prev_vx_error = {}
        self.prev_vy_error = {}

        # ---- MÓDULO DE PERTURBACIONES CONFIGURABLE ----
        self.perturbations = OrbitalPerturbations(
            dt=self.dt,
            srp_magnitude=1e-5,          
            drag_coefficient=5e-4,       
            thruster_error_period=50,    
            thruster_error_max=1e-3,     
            wind_magnitude=1e-3,         
            wind_slow_variation=0.003,   
            seed=self.seed
        )

        self.perturbation_scale = 0.0
        self.perturbation_increment = 5e-6   

        # ---- Target global y posiciones asignadas ----
        self.target_pos_global = np.array([0.0, 0.0], dtype=np.float32)
        self.target_ring_radius = 0.45
        self.tolerance_pos = 0.12
        self.success_speed = 0.08

        self.assigned_angles = np.array([
            (2 * np.pi * i / self.num_robots) for i in range(self.num_robots)
        ], dtype=np.float32)

        self.assigned_target_pos = {}

        # ---- Espacios ----
        self.obs_dim_per_robot = 4 + 2 * (self.num_robots - 1)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_robots * self.obs_dim_per_robot,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_robots, 2),
            dtype=np.float32
        )

        self.action_smoothing = 0.7
        self.ring_tolerance = 0.12
        self.last_action = np.zeros((n_agents, 2), dtype=np.float32)
        
        # Atributo persistente para el experimento de evaluación
        self.init_positions_this_episode = None

    # ===============================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.sim.model, self.sim.data)

        rng = np.random.default_rng(seed)
        self.current_step = 0

        margin = self.target_ring_radius + 0.15
        tx_center = rng.uniform(self.x_min + margin, self.x_max - margin)
        ty_center = rng.uniform(self.y_min + margin, self.y_max - margin)
        self.target_pos_global = np.array([tx_center, ty_center], dtype=np.float32)

        self.assigned_target_pos = {}
        for i, name in enumerate(self.sim.robot_names):
            ang = self.assigned_angles[i]
            tx = self.target_pos_global[0] + self.target_ring_radius * np.cos(ang)
            ty = self.target_pos_global[1] + self.target_ring_radius * np.sin(ang)
            self.assigned_target_pos[name] = np.array([tx, ty], dtype=np.float32)

        init_positions = []
        for i, name in enumerate(self.sim.robot_names):
            for _ in range(500):
                ix = rng.uniform(self.x_min, self.x_max)
                iy = rng.uniform(self.y_min, self.y_max)
                pos = np.array([ix, iy])

                dist_to_any_target = min([np.linalg.norm(pos - tp)
                                          for tp in self.assigned_target_pos.values()])

                dist_to_others = True
                for prev_pos in init_positions:
                    if np.linalg.norm(pos - np.array(prev_pos)) < 0.8:
                        dist_to_others = False
                        break

                if dist_to_any_target > 0.4 and dist_to_others:
                    init_positions.append([ix, iy])
                    break

            self.data.qpos[self.qadr_x[name]] = float(ix - self.base_x[name])
            self.data.qpos[self.qadr_y[name]] = float(iy - self.base_y[name])
            self.data.qpos[self.qadr_yaw[name]] = 0.0
            self.data.qvel[self.dadr_x[name]] = 0.0
            self.data.qvel[self.dadr_y[name]] = 0.0
            self.data.qvel[self.dadr_yaw[name]] = 0.0

            self.prev_vx_error[name] = 0.0
            self.prev_vy_error[name] = 0.0
            self.dvx[name] = 0.0
            self.dvy[name] = 0.0

        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.last_action = np.zeros((self.num_robots, 2), dtype=np.float32)

        self.init_positions_this_episode = np.array(init_positions, dtype=np.float32)

        return self._get_state(), {"init_pos": self.init_positions_this_episode}

    # ===============================
    def step(self, action):
        self.current_step += 1
        robot_names = self.sim.robot_names

        done = False
        r_coll_penalty = 0.0

        # --- Avanzar Currículum de Perturbaciones ---
        self.perturbation_scale = min(1.0, self.perturbation_scale + self.perturbation_increment)

        current_smoothing = self.action_smoothing
        if np.any(action * self.last_action < -0.2):
            current_smoothing = 0.10

        action = current_smoothing * self.last_action + (1 - current_smoothing) * action

        dist_old = {name: np.linalg.norm(self.data.xpos[self.body_id[name]][:2] - self.assigned_target_pos[name])
                    for name in robot_names}

        # --- PRE-CÁLCULO DE ASISTENCIA FÍSICA DE EMERGENCIA ---
        robot_positions = [self.data.xpos[self.body_id[n]][:2] for n in robot_names]
        emergency_brake = {name: np.zeros(2) for name in robot_names}

        if self.use_emergency_brake:
            for i, name_i in enumerate(robot_names):
                pos_i = robot_positions[i]
                for j, name_j in enumerate(robot_names):
                    if i != j:
                        rel_pos = robot_positions[j] - pos_i
                        dist = np.linalg.norm(rel_pos)
                        if dist < 0.38:
                            repulsion_dir = -rel_pos / (dist + 1e-6)
                            intensity = (0.38 - dist) / (0.38 - 0.26)
                            emergency_brake[name_i] += repulsion_dir * intensity * 2.5

        # --- PRE-CÁLCULO DE PERTURBACIONES ---
        pert_per_robot = {}
        for name in robot_names:
            vx0 = float(self.data.qvel[self.dadr_x[name]])
            vy0 = float(self.data.qvel[self.dadr_y[name]])
            vel0 = np.array([vx0, vy0])
            pert = self.perturbations.total_perturbation(
                velocity=vel0,
                step=self.current_step,
                use_srp=True,            
                use_drag=True,           
                use_thruster_error=True, 
                use_wind=True            
            )
            pert_scaled = pert * self.perturbation_scale
            pert_per_robot[name] = (
                pert_scaled[0] * self.mass[name],
                pert_scaled[1] * self.mass[name]
            )

        # --- DICCIONARIOS PARA EL EXPERIMENTO DEL TFG ---
        step_telemetry = {
            name: {"max_f_ctrl": 0.0, "max_f_brake": 0.0, "max_f_combined": 0.0}
            for name in robot_names
        }

        # 1. Bucle de control físico con asistencia integrada y perturbaciones
        for _ in range(5):
            for i, name in enumerate(robot_names):
                body_id = self.body_id[name]

                vx = float(self.data.qvel[self.dadr_x[name]])
                vy = float(self.data.qvel[self.dadr_y[name]])

                vx_des, vy_des = action[i] * self.v_cmd_max
                ex, ey = vx_des - vx, vy_des - vy
                alpha = 0.7
                raw_dex = (ex - self.prev_vx_error[name]) / self.dt
                raw_dey = (ey - self.prev_vy_error[name]) / self.dt
                self.dvx[name] = alpha * raw_dex + (1 - alpha) * self.dvx[name]
                self.dvy[name] = alpha * raw_dey + (1 - alpha) * self.dvy[name]

                Fx_ctrl = (self.kp_v * ex + self.kd_v * self.dvx[name] - self.kd_damping * vx)
                Fy_ctrl = (self.kp_v * ey + self.kd_v * self.dvy[name] - self.kd_damping * vy)

                self.prev_vx_error[name], self.prev_vy_error[name] = ex, ey

                F_norm = np.sqrt(Fx_ctrl**2 + Fy_ctrl**2) + 1e-8
                if F_norm > self.F_max:
                    Fx_ctrl = (Fx_ctrl / F_norm) * self.F_max
                    Fy_ctrl = (Fy_ctrl / F_norm) * self.F_max

                Fx_pert, Fy_pert = pert_per_robot[name]
                
                Fx_ext = Fx_pert + emergency_brake[name][0]
                Fy_ext = Fy_pert + emergency_brake[name][1]

                # --- MEDICIÓN CIENTÍFICA ---
                mag_ctrl = np.sqrt(Fx_ctrl**2 + Fy_ctrl**2)
                mag_brake = np.linalg.norm(emergency_brake[name])
                mag_combined = np.sqrt((Fx_ctrl + Fx_ext)**2 + (Fy_ctrl + Fy_ext)**2)

                if mag_ctrl > step_telemetry[name]["max_f_ctrl"]:
                    step_telemetry[name]["max_f_ctrl"] = mag_ctrl
                if mag_brake > step_telemetry[name]["max_f_brake"]:
                    step_telemetry[name]["max_f_brake"] = mag_brake
                if mag_combined > step_telemetry[name]["max_f_combined"]:
                    step_telemetry[name]["max_f_combined"] = mag_combined

                self.data.xfrc_applied[body_id][:2] = [Fx_ctrl + Fx_ext, Fy_ctrl + Fy_ext]
                self.data.qvel[self.dadr_yaw[name]] = 0.0

            mujoco.mj_step(self.sim.model, self.sim.data)

            for name in robot_names:
                self.data.xfrc_applied[self.body_id[name]][:] = 0.0

        # --- AJUSTE 1: Colisiones Reales ---
        collision_detected = False
        body_to_geom = {name: f"cube_{i+1}" for i, name in enumerate(robot_names)}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.dist < 0:
                g1 = self.model.geom(contact.geom1).name
                g2 = self.model.geom(contact.geom2).name
                for name_i, geom_i in body_to_geom.items():
                    for name_j, geom_j in body_to_geom.items():
                        if name_i != name_j:
                            if (g1 == geom_i and g2 == geom_j) or (g1 == geom_j and g2 == geom_i):
                                collision_detected = True
                                break
                if collision_detected:
                    break

        if collision_detected:
            r_coll_penalty = -2000.0
            done = True

        # --- AJUSTE 2: Escudo de Proximidad Sólido ---
        robot_positions = [self.data.xpos[self.body_id[n]][:2] for n in robot_names]
        r_proximity_penalty = 0.0
        proximity_threshold = 0.50  
        danger_threshold = 0.35     

        for i, name in enumerate(robot_names):
            pos_i = robot_positions[i]
            vel_i = np.array([self.data.qvel[self.dadr_x[name]], self.data.qvel[self.dadr_y[name]]])

            for j, pos_j in enumerate(robot_positions):
                if i != j:
                    rel_pos = pos_j - pos_i
                    dist_ij = np.linalg.norm(rel_pos)

                    if dist_ij < proximity_threshold:
                        severity = (proximity_threshold - dist_ij) / (proximity_threshold - 0.24)
                        r_proximity_penalty -= 5.0 * (severity ** 2)

                    if dist_ij < danger_threshold:
                        vel_j = np.array([self.data.qvel[self.dadr_x[robot_names[j]]],
                                          self.data.qvel[self.dadr_y[robot_names[j]]]])
                        rel_vel = vel_i - vel_j
                        rel_pos_unit = rel_pos / (dist_ij + 1e-6)
                        approach_speed = np.dot(rel_vel, rel_pos_unit)

                        if approach_speed > 0.02:
                            r_proximity_penalty -= (approach_speed * 12.0) / (dist_ij + 0.01)

        # --- 3. Recompensa de Navegación ---
        total_agent_reward = 0.0
        in_pos_count = 0
        all_stable = True
        for i, name in enumerate(robot_names):
            pos = robot_positions[i]
            target = self.assigned_target_pos[name]
            dist_now = np.linalg.norm(pos - target)
            vel = np.array([self.data.qvel[self.dadr_x[name]], self.data.qvel[self.dadr_y[name]]])
            speed = np.linalg.norm(vel)

            if dist_now < 0.06:
                r_dist = 15.0 * (1.0 - (speed / self.success_speed))
            elif dist_now < 0.4:
                r_dist = -2.0 * (dist_now ** 2)
            else:
                r_dist = -0.6 * dist_now

            r_prog = (dist_old[name] - dist_now) * 150.0

            if dist_now < self.tolerance_pos:
                in_pos_count += 1
                if speed > self.success_speed:
                    all_stable = False
            else:
                all_stable = False

            target_dir = target - pos
            target_dir_unit = target_dir / (np.linalg.norm(target_dir) + 1e-6)
            velocity_towards_target = np.dot(vel, target_dir_unit)

            r_velocity_alignment = 0.5 * velocity_towards_target
            total_agent_reward += (r_dist + r_prog + r_velocity_alignment)

        # --- 4. Suma Final ---
        action_delta_penalty = -1.5 * np.mean(np.square(action - self.last_action))

        reward_combined = (total_agent_reward / self.num_robots) + \
                          r_proximity_penalty + \
                          r_coll_penalty + \
                          action_delta_penalty

        if all_stable and in_pos_count == self.num_robots:
            reward_combined += 1000.0
            done = True

        self.last_action = action.copy()
        truncated = self.current_step >= self.max_steps

        info_dict = {
            "num_in_pos": in_pos_count,
            "init_pos": self.init_positions_this_episode,
            "actuator_telemetry": step_telemetry
        }

        return self._get_state(), float(reward_combined), bool(done), bool(truncated), info_dict

    # ===============================
    def _get_state(self):
        robot_names = self.sim.robot_names
        positions = np.array([self.data.xpos[self.body_id[name]][:2].copy() for name in robot_names])
        velocities = np.array([[self.data.qvel[self.dadr_x[name]], self.data.qvel[self.dadr_y[name]]]
                               for name in robot_names])

        states = []
        for i, name in enumerate(robot_names):
            pos = positions[i]
            vel = velocities[i]
            target = self.assigned_target_pos[name]

            delta = target - pos
            state_i = [
                np.clip(delta[0] / self.Lx, -1.0, 1.0),
                np.clip(delta[1] / self.Ly, -1.0, 1.0),
                np.clip(vel[0] / self.v_max, -1.0, 1.0),
                np.clip(vel[1] / self.v_max, -1.0, 1.0)
            ]

            other_deltas = []
            for j in range(self.num_robots):
                if i == j:
                    continue
                d_vec = positions[j] - pos
                dist = np.linalg.norm(d_vec)
                other_deltas.append((dist, d_vec))

            other_deltas.sort(key=lambda x: x[0])

            for dist, d_vec in other_deltas:
                state_i.append(float(np.clip(d_vec[0] / 1.5, -1.0, 1.0)))
                state_i.append(float(np.clip(d_vec[1] / 1.5, -1.0, 1.0)))

            states.append(np.array(state_i, dtype=np.float32))

        return np.concatenate(states)

    def render(self, mode="human"):
        pass