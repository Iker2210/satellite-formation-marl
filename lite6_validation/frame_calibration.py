import numpy as np
import mujoco
import itertools
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI
from lite6_new_env import Lite6Env
import time

# Ángulos REALES del robot ahora
ANGULOS = [0, 0, 1.570796, 0, 0, 0]

# Sim con los mismos ángulos
env = Lite6Env()
env.reset()
env.data.qpos[7:13] = np.array(ANGULOS)
env.data.ctrl[0:6] = np.array(ANGULOS)
env.data.qvel[:] = 0
mujoco.mj_forward(env.model, env.data)
tcp_sim_world = env._get_ee_pos()
R_tcp_sim = env.data.site_xmat[env.ee_site_id].reshape(3, 3)

# Real
arm = XArmAPI('192.168.50.5')
time.sleep(1)
tcp_real_mm = np.array(arm.position[:3])
roll, pitch, yaw = arm.position[3], arm.position[4], arm.position[5]
R_tcp_real = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
arm.disconnect()

print(f"TCP sim (m):   {tcp_sim_world}")
print(f"TCP real (mm): {tcp_real_mm}")
print(f"\nR_tcp_sim:\n{R_tcp_sim}")
print(f"\nR_tcp_real:\n{R_tcp_real}")

# Encontrar M que mejor relaciona TODO (rotación + posición con offset)
OFFSET = np.array([0.0, 0.0, 0.22])
tcp_real_m = tcp_real_mm / 1000.0

mejor = None
mejor_err = 1e9
for perm in itertools.permutations([0, 1, 2]):
    for signos in itertools.product([1, -1], repeat=3):
        M = np.zeros((3, 3))
        for fila, (col, s) in enumerate(zip(perm, signos)):
            M[fila, col] = s

        err_rot = np.abs(M @ R_tcp_real - R_tcp_sim).max()
        err_pos = np.abs(M @ tcp_real_m + OFFSET - tcp_sim_world).max()
        err = err_rot + err_pos

        if err < mejor_err:
            mejor_err = err
            mejor = (M.copy(), err_rot, err_pos)

M, err_rot, err_pos = mejor
print(f"\n=== MEJOR M ===")
print(M)
print(f"Error rotación: {err_rot:.4f}")
print(f"Error posición: {err_pos*1000:.1f} mm")
print(f"\nM @ R_tcp_real:\n{M @ R_tcp_real}")
print(f"M @ tcp_real + offset: {M @ tcp_real_m + OFFSET}")
print(f"Esperado en sim:       {tcp_sim_world}")