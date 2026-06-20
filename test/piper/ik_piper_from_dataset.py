#!/usr/bin/env python3
"""MuJoCo viewer for the Piper arm."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer

_PIPER_XML = Path(__file__).resolve().parents[2] / "assets" / "piper" / "piper.xml"


def _init_actuator_ctrl(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Align ``data.ctrl`` with the current joint positions."""
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        qpos_adr = model.jnt_qposadr[joint_id]
        data.ctrl[act_id] = data.qpos[qpos_adr]


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(_PIPER_XML))
    data = mujoco.MjData(model)

    # joint1 tiene damping=300 en el XML, que lo deja muy sobreamortiguado
    # (amortiguamiento crítico ≈ 2 con kp=10000, I≈1e-4).  Reducirlo aquí
    # lo hace responder mucho más rápido sin editar el asset.
    joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    model.dof_damping[model.jnt_dofadr[joint1_id]] = 10.0

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    _init_actuator_ctrl(model, data)
    mujoco.mj_forward(model, data)

    print("\nVisor MuJoCo abierto.")
    print("  - Panel Control (derecha): mueve los actuadores del brazo.")
    print("  - Panel Joint (izquierda): pose cinemática directa con simulación en pausa.")
    print("Cierra la ventana para salir.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()
            sleep_time = model.opt.timestep - (time.monotonic() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
