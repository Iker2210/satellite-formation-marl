# Satellite Formation Control via Multi-Agent Reinforcement Learning

**BSc Thesis — Robotics Engineering, University of Alicante (2026)**  
**Author:** Iker Alaminos Hernández  
**Supervisor:** Dr. Jorge Pomares Baeza

---

## Overview

This repository contains the full implementation of a distributed formation-flight controller for a swarm of satellites using **Multi-Agent Reinforcement Learning (MARL)**. The system is trained entirely in simulation (MuJoCo + Gymnasium) and validated on physical hardware via **Sim-to-Real transfer** on a Ufactory Lite6 robotic arm over a pneumatic air-bearing platform.

The core research question: *can a group of satellites learn to autonomously coordinate and maintain a stable ring formation under realistic orbital perturbations — without explicit inter-agent communication?*

**Answer: yes, with 99.1% mission success rate.**

---

## Key Results

| Metric | Value |
|--------|-------|
| Mission success rate (4 agents, 1000 episodes) | **99.1%** |
| Collision rate | 0.9% |
| Mean positioning error | ~5 mm |
| Terminal velocity | ~0.014 m/s |
| PPO vs PID (safety layer active) | **99.1% vs 85.9%** |
| PPO vs PID (safety layer off) | **80.7% vs 24.3%** |
| Safety layer activation (PPO vs PID) | **2.46% vs 37.19%** (×15 factor) |
| Impulse recovery rate (0.1–2.0 N, 120 episodes) | **100%** |
| Lite6 Sim-to-Real success | 3/3 targets, <2 mm calibration error |

---

## System Architecture

```
Policy (PPO/SAC — CTDE)
        │
        ▼
   Action [-1, 1]
        │
        ▼
  Force scaling / PD
  (Fx, Fy, τ per agent)
        │
        ▼
   MuJoCo Physics
   Δt = 0.01 s
        │
   + Perturbations ──► J2 oblateness · Atmospheric drag
                        Solar radiation pressure · Thruster noise
        │
        ▼
  Normalised observation
  (own state + relative neighbours)
        │
        ▼
   Reward signal
   (progress · stability · collision avoidance · energy)
```

**Training paradigm:** Centralised Training / Decentralised Execution (CTDE) with parameter sharing across agents.

---

## Orbital Perturbations Modelled

- **J2 oblateness** — Earth's equatorial bulge effect on orbital dynamics
- **Atmospheric drag** — altitude-dependent drag in Low Earth Orbit (LEO)
- **Solar Radiation Pressure (SRP)** — photon pressure on satellite surfaces
- **Thruster noise** — actuation error modelling microthruster imprecision
- **Orbital wind** — residual aerodynamic effects

---

## Repository Structure

```
satellite-formation-marl/
│
├── env/
│   ├── satellite_env.py        # Gymnasium environment — satellite formation
│   ├── lite6_new_env.py        # Gymnasium environment — Ufactory Lite6
│   └── physics.py              # Orbital perturbations module
│
├── src/
│   ├── multiagente/            # Training & evaluation scripts (1–4 agents)
│   └── lite6/                  # Lite6 training & evaluation scripts
│
├── assets/
│   ├── robot_cube.xml          # MuJoCo model — 4-satellite formation
│   └── lite6_airbearings.xml   # MuJoCo model — Ufactory Lite6 on air bearings
│
├── models/
│   └── README.md               # Models not included — see README for details
│
├── lite6_validation/
│   ├── frame_calibration.py       # Sim-to-Real frame transformation
│   ├── sim_to_real_horizontal.py  # Real hardware inference — horizontal config (air bearings)
│   ├── sim_to_real_vertical.py    # Real hardware inference — vertical config
│   ├── base_drift_analysis.py     # Base drift and orientation analysis
│   └── env_diagnostic.py          # Environment sanity checks and MuJoCo viewer test
│
├── logs/
│   └── README.md               # Logs not included — regenerated on training
│
└── README.md
```

> **Note:** `lite6_airbearings.xml` requires Ufactory Lite6 mesh files (`.stl`) not included here. Download them from the [MuJoCo Menagerie repository](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_lite6) and place them in `assets/visual/` and `assets/collision/`.

---

## Progressive Training Curriculum

The system was trained following a curriculum learning approach, scaling from a single agent to a 4-satellite formation:

| Phase | Description | Commit |
|-------|-------------|--------|
| 1 agent | Individual policy under orbital perturbations | `master` |
| 2 agents | First stable cooperative model | `4633865` |
| 3 agents | Stable 3-agent policy with angular distribution | `52a6995` |
| 4 agents | Final model — full perturbations | `1567016` |
| Lite6 | Manipulator policy on floating base | `c5badca` |

---

## Emergent Behaviours

Three cooperative regimes emerged **without being explicitly programmed**, arising purely from the reward design and local observation structure:

1. **Immediate compensation** (≤0.6 N impulse) — perturbed agent self-corrects without affecting neighbours
2. **Local active recovery** (1.0 N) — perturbed agent recovers in ~5 s independently
3. **Cooperative repositioning** (1.5–2.0 N) — neighbouring agents temporarily reposition to avoid collision, then return to formation

---

## Sim-to-Real Transfer — Ufactory Lite6

The Lite6 manipulator was used as a terrestrial analogue of a satellite-mounted robotic arm under microgravity conditions (horizontal configuration on pneumatic air bearings).

- Policy trained entirely in MuJoCo simulation
- Transferred to real hardware with **zero fine-tuning**
- Calibration error between simulated and real frames: **< 2 mm**
- Inference loop running at **100 Hz** on real hardware
- Success on all 3 evaluated targets with reproducible behaviour across runs

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Physics simulation | MuJoCo |
| RL environment | Gymnasium (Farama Foundation) |
| RL algorithms | Stable-Baselines3 (PPO, SAC) |
| Language | Python |
| Monitoring | TensorBoard |
| Hardware | Ufactory Lite6 + pneumatic air-bearing platform |

---

## Video Documentation

Full development playlist (phases 1–7 + Sim-to-Real):  
📺 [YouTube Playlist](http://youtube.com/playlist?list=PLoc0NGBSnR_72sjY87do1GFE11iu5CG4F)

---

## About

This project is part of **IA Robots** — a personal research platform documenting original work in autonomous and space robotics.  
🌐 [iarobots.tech](https://iarobots.tech)  
🔗 [LinkedIn](https://linkedin.com/in/iker-alaminos-hernandez-iarobots)

> The original development repository, including the full commit history across all training phases, is available at [TFG-Reinforcement-Learning](https://github.com/Iker2210/TFG-Reinforcement-Learning).
