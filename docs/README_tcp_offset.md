# Calibración Controller → TCP

Calcula la punta física del HandUMI desde el origen de tracking Meta:

```text
T_world_tcp = T_world_controller @ T_controller_tcp
```

Piper usa `configs/calibration/meta_controller_tcp.yaml`. Recalibrar solo si
cambia el montaje físico. El pivot calibra traslación; conservar los
cuaterniones CAD actuales. La calibración de apertura está en
[README_gripper_width.md](README_gripper_width.md).

## 1. Preparación

- Headset y aplicación Meta conectados.
- Ambos controladores encendidos, despiertos y visibles.
- Punta apoyada en una hendidura o esquina firme.
- Un directorio nuevo para cada intento.

Durante los 25 segundos, mantener la punta inmóvil y mover el mango lentamente:

1. Adelante y atrás.
2. Izquierda y derecha.
3. Giro horario y antihorario.
4. Inclinaciones diagonales.

## 2. Izquierda

```bash
LEFT_RUN="outputs/tcp_pivot_left_$(date +%Y%m%d_%H%M%S)"
uv run handumi-record \
  --device meta \
  --skip-feetech \
  --only-left-camera \
  --repo-id local/tcp_pivot_left \
  --output-dir "$LEFT_RUN" \
  --task "tcp pivot left" \
  --num-episodes 1 \
  --episode-time-s 25 \
  --tracking-loss-timeout-s 3 \
  --no-sounds
```

Presionar `ENTER` con la punta ya inmovilizada. Confirmar `Episode 1 saved`.

```bash
uv run handumi-calibrate-tcp-offset pivot \
  --device meta \
  --side left \
  --parquet "$LEFT_RUN/data/chunk-000/file-000.parquet" \
  --episode 0 \
  --output outputs/calibration/meta_controller_tcp_candidate.yaml
```

## 3. Derecha

```bash
RIGHT_RUN="outputs/tcp_pivot_right_$(date +%Y%m%d_%H%M%S)"
uv run handumi-record \
  --device meta \
  --skip-feetech \
  --only-right-camera \
  --repo-id local/tcp_pivot_right \
  --output-dir "$RIGHT_RUN" \
  --task "tcp pivot right" \
  --num-episodes 1 \
  --episode-time-s 25 \
  --tracking-loss-timeout-s 3 \
  --no-sounds
```

```bash
uv run handumi-calibrate-tcp-offset pivot \
  --device meta \
  --side right \
  --parquet "$RIGHT_RUN/data/chunk-000/file-000.parquet" \
  --episode 0 \
  --output outputs/calibration/meta_controller_tcp_candidate.yaml
```

El segundo cálculo conserva `left` y agrega `right`.

## 4. Aceptación

```text
RMS < 0.50 cm
máximo < 1.00 cm recomendado
condition < 500
```

RMS alto: punta deslizada. `condition` alto: poca variedad de giros. Repetir
en otro directorio.

```bash
uv run handumi-calibrate-tcp-offset inspect \
  outputs/calibration/meta_controller_tcp_candidate.yaml
```

## 5. Verificación

```bash
JAX_PLATFORMS=cpu uv run handumi-teleop-sim \
  --device meta \
  --robot piper \
  --controller-tcp-calibration outputs/calibration/meta_controller_tcp_candidate.yaml
```

- Girar alrededor de una punta quieta: TCP simulado casi inmóvil.
- Tocar el mismo punto con ambas puntas: estelas coincidentes.
- Tocar la mesa: ambos TCP cerca de `z=0`.

Replay de comprobación:

```bash
JAX_PLATFORMS=cpu uv run handumi-replay-in-sim \
  --repo-id your-name/handumi-demo \
  --dataset-root outputs/handumi-demo \
  --episode 0 \
  --robot piper \
  --controller-tcp-calibration outputs/calibration/meta_controller_tcp_candidate.yaml
```

## 6. Promoción

No copiar el candidato completo: el pivot no calibra orientación. Conservar los
cuaterniones oficiales y proyectar las traslaciones medidas sobre la simetría:

```text
x = (left.x + right.x) / 2
y = (left.y - right.y) / 2
z = (left.z + right.z) / 2

left.position  = [x,  y, z]
right.position = [x, -y, z]
```

Actualizar solo `position` en
`configs/calibration/meta_controller_tcp.yaml`. Piper la selecciona
automáticamente. `--use-dataset-tcp-calibration` queda únicamente para
reproducir un snapshot histórico.

```bash
uv run pytest -q tests/tracking/test_transforms.py \
  tests/scripts/test_replay_in_sim.py
```
