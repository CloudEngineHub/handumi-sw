#!/usr/bin/env python3
"""
Visualiza la trayectoria de los controladores Pico (izquierdo y derecho)
a partir de un dataset en formato LeRobot almacenado localmente.

Uso:
    uv run python test/visualize_pico_controller.py datasets/my_dataset
    uv run python test/visualize_pico_controller.py datasets/my_dataset --episode 0
    uv run python test/visualize_pico_controller.py datasets/my_dataset --speed 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation


# ── Carga de datos ─────────────────────────────────────────────────────────────

def load_dataset(root: Path) -> pd.DataFrame:
    """Lee todos los parquet del dataset y los concatena ordenados por índice."""
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        sys.exit(f"[ERROR] No se encontraron archivos .parquet en {root / 'data'}")

    frames = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("index", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def extract_poses(df: pd.DataFrame, episode: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna (timestamps, left_poses, right_poses) para el episodio indicado.
    Cada pose tiene shape (N, 7): [x, y, z, qx, qy, qz, qw].
    """
    if episode is not None:
        df = df[df["episode_index"] == episode]
        if df.empty:
            sys.exit(f"[ERROR] El episodio {episode} no existe en el dataset.")
    else:
        # Tomar el primer episodio disponible
        episode = int(df["episode_index"].iloc[0])
        df = df[df["episode_index"] == episode]

    left  = np.stack(df["observation.pico.left_controller_pose"].values)   # (N, 7)
    right = np.stack(df["observation.pico.right_controller_pose"].values)  # (N, 7)
    ts    = df["timestamp"].values.astype(float).ravel()

    return ts, left, right


# ── Geometría ──────────────────────────────────────────────────────────────────

def quat_to_axes(q: np.ndarray, scale: float = 0.05) -> np.ndarray:
    """
    Convierte cuaternión [qx, qy, qz, qw] en tres vectores de ejes (X, Y, Z)
    escalados. Retorna array (3, 3): filas = ejes X, Y, Z en coordenadas mundo.
    """
    R = Rotation.from_quat(q).as_matrix()  # [qx, qy, qz, qw] → matriz 3x3
    return R * scale  # broadcast: cada columna escalada


def axis_limits(left: np.ndarray, right: np.ndarray, margin: float = 0.1):
    all_pos = np.vstack([left[:, :3], right[:, :3]])
    lo = all_pos.min(axis=0) - margin
    hi = all_pos.max(axis=0) + margin
    return lo, hi


# ── Animación ──────────────────────────────────────────────────────────────────

TRIAD_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]  # X=rojo, Y=verde, Z=azul


def build_triad_lines(ax, pos: np.ndarray, axes_vecs: np.ndarray):
    """Crea las tres líneas del triedro de orientación."""
    lines = []
    for i, color in enumerate(TRIAD_COLORS):
        end = pos + axes_vecs[:, i]
        ln, = ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]],
                      color=color, lw=1.5, alpha=0.9)
        lines.append(ln)
    return lines


def update_triad(lines, pos: np.ndarray, axes_vecs: np.ndarray):
    for i, ln in enumerate(lines):
        end = pos + axes_vecs[:, i]
        ln.set_data([pos[0], end[0]], [pos[1], end[1]])
        ln.set_3d_properties([pos[2], end[2]])


def run_animation(ts: np.ndarray, left: np.ndarray, right: np.ndarray,
                  speed: float, title: str) -> None:

    lo, hi = axis_limits(left, right)
    N = len(ts)
    interval_ms = max(10, int((1000 / 30) / speed))  # ~30 fps real, escalado

    fig = plt.figure(figsize=(10, 7), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d", facecolor="#16213e")
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)

    ax.set_title(title, color="white", fontsize=11, pad=8)
    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        spine.pane.fill = False
        spine.pane.set_edgecolor("#0f3460")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    ax.set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Y (m)", color="#aaaaaa", fontsize=8)
    ax.set_zlabel("Z (m)", color="#aaaaaa", fontsize=8)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])

    # Trayectorias completas (translúcidas)
    ax.plot(left[:, 0],  left[:, 1],  left[:, 2],
            color="#5b9bd5", lw=0.6, alpha=0.3, zorder=1)
    ax.plot(right[:, 0], right[:, 1], right[:, 2],
            color="#e8a87c", lw=0.6, alpha=0.3, zorder=1)

    # Trayectoria recorrida (se irá rellenando)
    trail_l, = ax.plot([], [], [], color="#5b9bd5", lw=1.5, alpha=0.7, zorder=2)
    trail_r, = ax.plot([], [], [], color="#e8a87c", lw=1.5, alpha=0.7, zorder=2)

    # Puntos actuales
    dot_l, = ax.plot([], [], [], "o", color="#74b9ff", ms=8, zorder=5)
    dot_r, = ax.plot([], [], [], "o", color="#fdcb6e", ms=8, zorder=5)

    # Triedros de orientación (inicializados en frame 0)
    axes_l0 = quat_to_axes(left[0,  3:])
    axes_r0 = quat_to_axes(right[0, 3:])
    triad_l = build_triad_lines(ax, left[0,  :3], axes_l0)
    triad_r = build_triad_lines(ax, right[0, :3], axes_r0)

    # Indicador de tiempo
    time_txt = ax.text2D(0.02, 0.97, "", transform=ax.transAxes,
                         color="white", fontsize=9, va="top")

    # Leyenda
    legend_items = [
        Line2D([0], [0], color="#74b9ff", lw=2, label="Left controller"),
        Line2D([0], [0], color="#fdcb6e", lw=2, label="Right controller"),
        Line2D([0], [0], color="#e74c3c", lw=1.5, label="Axis X"),
        Line2D([0], [0], color="#2ecc71", lw=1.5, label="Axis Y"),
        Line2D([0], [0], color="#3498db", lw=1.5, label="Axis Z"),
    ]
    leg = ax.legend(handles=legend_items, loc="upper right",
                    facecolor="#0f3460", edgecolor="#aaaaaa",
                    labelcolor="white", fontsize=8)

    def init():
        trail_l.set_data([], [])
        trail_l.set_3d_properties([])
        trail_r.set_data([], [])
        trail_r.set_3d_properties([])
        dot_l.set_data([], [])
        dot_l.set_3d_properties([])
        dot_r.set_data([], [])
        dot_r.set_3d_properties([])
        return trail_l, trail_r, dot_l, dot_r, *triad_l, *triad_r

    def update(frame: int):
        i = frame + 1  # incluye el frame actual

        # Trails
        trail_l.set_data(left[:i, 0], left[:i, 1])
        trail_l.set_3d_properties(left[:i, 2])
        trail_r.set_data(right[:i, 0], right[:i, 1])
        trail_r.set_3d_properties(right[:i, 2])

        # Puntos actuales
        pl = left[frame, :3]
        pr = right[frame, :3]
        dot_l.set_data([pl[0]], [pl[1]])
        dot_l.set_3d_properties([pl[2]])
        dot_r.set_data([pr[0]], [pr[1]])
        dot_r.set_3d_properties([pr[2]])

        # Triedros
        update_triad(triad_l, pl, quat_to_axes(left[frame,  3:]))
        update_triad(triad_r, pr, quat_to_axes(right[frame, 3:]))

        t = ts[frame]
        time_txt.set_text(f"t = {t:.2f} s  |  frame {frame + 1}/{N}")

        return trail_l, trail_r, dot_l, dot_r, *triad_l, *triad_r, time_txt

    ani = animation.FuncAnimation(
        fig, update, frames=N, init_func=init,
        interval=interval_ms, blit=False, repeat=True,
    )

    plt.tight_layout()
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Ruta raíz del dataset (p. ej. datasets/my_dataset)",
    )
    parser.add_argument(
        "--episode", "-e",
        type=int,
        default=None,
        help="Índice del episodio a visualizar (por defecto: el primero disponible).",
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Factor de velocidad de la animación (1.0 = tiempo real, 2.0 = doble). Default: 1.0",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.dataset.resolve()

    if not root.exists():
        sys.exit(f"[ERROR] No existe el directorio: {root}")

    # Leer info del dataset
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    fps = info.get("fps", 30)
    total_episodes = info.get("total_episodes", "?")

    print(f"Dataset : {root.name}")
    print(f"FPS     : {fps}  |  Episodios totales: {total_episodes}")

    df = load_dataset(root)
    ts, left, right = extract_poses(df, args.episode)

    ep_label = args.episode if args.episode is not None else int(df["episode_index"].iloc[0])
    title = f"{root.name}  —  Episode {ep_label}  ({len(ts)} frames @ {fps} fps)"
    print(f"Frames  : {len(ts)}  |  Visualizando episodio {ep_label}")

    run_animation(ts, left, right, speed=args.speed, title=title)


if __name__ == "__main__":
    main()
