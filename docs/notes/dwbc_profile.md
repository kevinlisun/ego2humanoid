# Decoupled WBC capability profile (L.3)

Measured by `scripts/profile_dwbc.py` on `g1_29dof_with_hand_rev_1_0.xml` (flat floor, 3 seeds per walk case, 6 s each); raw data in `configs/dwbc_profile.json`.

- No falls in any case. Forward speed is about 0.8x the commanded vx (0.4 -> 0.33 m/s, 0.6 -> 0.51 m/s); backward -0.3 -> -0.28 m/s.
- Pure yaw command 0.3 rad/s -> about 0.27-0.31 rad/s; lateral vy 0.2 -> about 0.13 m/s.
- Combined forward + yaw (0.3, 0, 0.3) loses forward speed (0.14 m/s) and drifts sideways (0.17 m/s): the planner should turn in place, then walk straight.
- Stop from 0.4 m/s: 0.08 m overshoot, settled in 0.55 s.
- Stable standing for height 0.5-0.8 m and waist pitch -0.3..0.5, roll 0.2, yaw 0.5 rad (`rpy_rad` command). Pelvis tracks height up to ~0.77 m. These form the supported posture set for gaze control (the G1 has no neck).
- The asset needed joint armature 0.01 / damping 0.001 / frictionloss 0.1; `DwbcRobot` applies them.
