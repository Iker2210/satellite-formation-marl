"""
PRUEBA DE MÚLTIPLES TARGETS EN EL PLANO Y≈0
=============================================
El modelo solo controla J2/J3/J5, que en el robot vertical mueven
el TCP en su plano vertical (ejes X y Z reales). El eje Y (lateral)
es inalcanzable porque requiere J1.

Por eso probamos targets con Y≈0 (plano del brazo). El error en Y
será siempre ~2-3 mm (HOME tiene Y=2.4) y es esperado/despreciable.
El "éxito" se mide por la precisión en X y Z.
"""

import time
import numpy as np
from xarm.wrapper import XArmAPI
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from scipy.spatial.transform import Rotation as R
from lite6_new_env import Lite6Env

# ============ CONFIGURATION — update before running ============
IP_ROBOT = '192.168.50.5'        # Update to your robot's IP
MODEL_PATH = "models/run_51/best_model.zip"      # Update to your model path
STATS_PATH = "models/run_51/vec_normalize.pkl"   # Update to your stats path

HOME_ANGULOS_RAD = [0.0, 0.0, 2.356194, 0.0, 0.0, 0.0]
JOINTS_ACTIVOS = [1, 2, 4]
LIMITES = {1: (-2.6, 2.6), 2: (0.5, 2.36), 4: (-2.5, 2.5)}

FACTOR_PASO = 0.002
ALPHA = 0.3
RESYNC_THRESHOLD = 0.05
TOL_LLEGADA_M = 0.06
MAX_ITER_POR_TARGET = 3000
FREQ_HZ = 100

R_real_to_sim = np.array([
    [0,  0, -1],
    [1,  0,  0],
    [0, -1,  0],
], dtype=np.float64)


def real_a_sim_m(p_real_mm):
    return R_real_to_sim @ (np.array(p_real_mm) / 1000.0)


def construir_obs(target_real_mm, arm):
    base_quat = np.array([0.7071068, 0, -0.7071068, 0])
    com_rel = np.array([-0.1, 0.0, 0.15])
    base_vel = np.zeros(6)

    code_q, q_real = arm.get_servo_angle(is_radian=True)
    code_v, states = arm.get_joint_states(is_radian=True)
    q_real = np.array(q_real[:6]) if code_q == 0 else np.zeros(6)
    v_real = np.array(states[1][:6]) if (code_v == 0 and states) else np.zeros(6)
    arm_obs = np.array([q_real[1], q_real[2], q_real[4],
                        v_real[1], v_real[2], v_real[4]])

    code_p, ee_raw = arm.get_position(is_radian=False)
    if code_p != 0:
        raise RuntimeError("Fallo leyendo TCP")
    ee_pos_real_mm = np.array(ee_raw[:3])
    roll, pitch, yaw = ee_raw[3], ee_raw[4], ee_raw[5]

    ee_pos_sim_m = real_a_sim_m(ee_pos_real_mm)
    target_sim_m = real_a_sim_m(target_real_mm)

    R_tcp_in_robot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    R_tcp_in_sim = R_real_to_sim @ R_tcp_in_robot

    error_global_sim = target_sim_m - ee_pos_sim_m
    error_local = R_tcp_in_sim.T @ error_global_sim

    dist = np.linalg.norm(error_local)
    compass_local = error_local / (dist + 1e-6)

    obs = np.concatenate([
        base_quat, com_rel, base_vel,
        arm_obs, error_local, compass_local
    ]).astype(np.float32)

    return obs, ee_pos_real_mm, dist


def ir_a_home(arm, confirmar=False):
    """Vuelve a HOME en modo 0. Sin confirmación por defecto (loop automático)."""
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.3)
    if confirmar:
        input("⏸  Enter para ir a HOME...")
    ret = arm.set_servo_angle(angle=HOME_ANGULOS_RAD,
                              is_radian=True, speed=0.3, wait=True)
    ok = (ret == 0)
    if ok:
        print(f"  ✅ En HOME. TCP: {[round(v,1) for v in arm.position[:3]]} mm")
    else:
        print(f"  ❌ Falló ir a HOME (ret={ret})")
    return ok


def alcanzar_target(arm, model, venv, target_xyz_mm):
    """
    Intenta alcanzar un target. Devuelve un dict con los resultados.
    El robot debe estar en mode=1 antes de llamar.
    """
    _, q_inicial = arm.get_servo_angle(is_radian=True)
    posicion_objetivo = np.array(q_inicial[:6])

    mejor_dist = float('inf')
    mejor_tcp = None
    iteracion = 0
    llego = False

    while iteracion < MAX_ITER_POR_TARGET:
        if arm.has_error or arm.state == 4:
            print(f"  ❌ Hardware error: {arm.error_code}")
            break

        _, q_real_now = arm.get_servo_angle(is_radian=True)
        if q_real_now:
            q_real_now = np.array(q_real_now[:6])
            if np.abs(posicion_objetivo - q_real_now).max() > RESYNC_THRESHOLD:
                posicion_objetivo = q_real_now.copy()

        obs_raw, ee_pos_mm, dist = construir_obs(target_xyz_mm, arm)
        obs_norm = venv.normalize_obs(obs_raw)
        action, _ = model.predict(obs_norm, deterministic=True)

        if dist < mejor_dist:
            mejor_dist = dist
            mejor_tcp = ee_pos_mm.copy()

        if dist < TOL_LLEGADA_M:
            llego = True
            mejor_tcp = ee_pos_mm.copy()
            break

        suavizado = min(1.0, iteracion / 100.0)
        target_proximo = posicion_objetivo.copy()
        for i, j_idx in enumerate(JOINTS_ACTIVOS):
            delta = float(action[i]) * FACTOR_PASO * suavizado
            lo, hi = LIMITES[j_idx]
            target_proximo[j_idx] = np.clip(target_proximo[j_idx] + delta, lo, hi)

        for j_idx in JOINTS_ACTIVOS:
            posicion_objetivo[j_idx] = (ALPHA * target_proximo[j_idx]
                                        + (1 - ALPHA) * posicion_objetivo[j_idx])

        arm.set_servo_angle_j(angles=posicion_objetivo.tolist(), is_radian=True)

        if iteracion % 200 == 0:
            print(f"  Iter {iteracion:4d} | dist={dist*1000:5.1f} mm "
                  f"| TCP=[{ee_pos_mm[0]:.0f}, {ee_pos_mm[1]:.0f}, {ee_pos_mm[2]:.0f}]")

        iteracion += 1
        time.sleep(1.0 / FREQ_HZ)

    # Descomponer el error final en X, Z (plano) y Y (inalcanzable)
    err_vec = mejor_tcp - np.array(target_xyz_mm)
    err_plano = np.sqrt(err_vec[0]**2 + err_vec[2]**2)  # X y Z
    err_lateral = abs(err_vec[1])                        # Y

    return {
        'target': target_xyz_mm,
        'llego': llego,
        'iteraciones': iteracion,
        'tcp_final': mejor_tcp,
        'dist_total': mejor_dist * 1000,
        'err_plano_xz': err_plano,
        'err_lateral_y': err_lateral,
    }


# =================== MAIN ===================
if __name__ == "__main__":
    # ===== TARGETS A PROBAR (frame real, mm), todos con Y≈0 =====
    # Variamos X (alcance frontal) y Z (altura). Y=0 = plano del brazo.
    # HOME está en X=143, Z=708. Probamos puntos alrededor.
    TARGETS = [
        [250, 0, 550],
        [300, 0, 450],
        [350, 0, 400],
        [300, 0, 600],
        [200, 0, 500],
        [400, 0, 450],
    ]

    arm = XArmAPI(IP_ROBOT)
    time.sleep(2)
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)

    # Ir a HOME inicial con confirmación
    print("→ HOME inicial")
    if not ir_a_home(arm, confirmar=True):
        arm.disconnect()
        exit(1)

    # Cargar modelo
    print("\nCargando modelo RL...")
    def make_env(): return Lite6Env()
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(STATS_PATH, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(MODEL_PATH, device='cpu')
    print("Modelo cargado.")

    print("\n" + "="*60)
    print(f"⚠️  Voy a probar {len(TARGETS)} targets. Mano en paro de emergencia.")
    print("="*60)
    input("⏸  Enter para empezar la batería de pruebas...")

    resultados = []
    try:
        for idx, target in enumerate(TARGETS):
            print(f"\n{'='*60}")
            print(f"TARGET {idx+1}/{len(TARGETS)}: {target} mm")
            print(f"{'='*60}")

            # Modo tiempo real para la inferencia
            arm.set_mode(1)
            arm.set_state(0)
            time.sleep(1.0)

            res = alcanzar_target(arm, model, venv, target)
            resultados.append(res)

            estado = "✅ LLEGÓ" if res['llego'] else "⚠️  no llegó"
            print(f"  {estado} | iter={res['iteraciones']} | "
                  f"dist total={res['dist_total']:.1f} mm | "
                  f"err plano XZ={res['err_plano_xz']:.1f} mm | "
                  f"err lateral Y={res['err_lateral_y']:.1f} mm")

            # Volver a HOME en modo 0 antes del siguiente target
            print("  Volviendo a HOME...")
            ir_a_home(arm, confirmar=False)

    except KeyboardInterrupt:
        print("\n🛑 Detenido manualmente.")
    finally:
        arm.set_state(4)
        time.sleep(0.3)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.3)
        arm.disconnect()
        print("✅ Desconectado.")

    # ===== TABLA RESUMEN =====
    print("\n\n" + "="*78)
    print("RESUMEN DE RESULTADOS")
    print("="*78)
    print(f"{'Target (mm)':<22} {'Llegó':<8} {'Iter':<7} "
          f"{'Dist tot':<10} {'Err XZ':<10} {'Err Y':<8}")
    print("-"*78)
    for r in resultados:
        t = r['target']
        t_str = f"[{t[0]}, {t[1]}, {t[2]}]"
        llego_str = "Sí" if r['llego'] else "No"
        print(f"{t_str:<22} {llego_str:<8} {r['iteraciones']:<7} "
              f"{r['dist_total']:<10.1f} {r['err_plano_xz']:<10.1f} "
              f"{r['err_lateral_y']:<8.1f}")
    print("="*78)
    print("Err XZ = error en el plano del brazo (lo que el modelo SÍ puede controlar)")
    print("Err Y  = error lateral (inalcanzable con J2/J3/J5, esperado ~2-3 mm)")