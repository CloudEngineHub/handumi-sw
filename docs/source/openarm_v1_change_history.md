# Historial de cambios de OpenArm v1 y teleoperacion

Ultima modificacion: 2026-07-15 18:32:28 -05 -0500

Este documento registra el trabajo realizado despues del commit
`77fe91e feat(openarm): add real robot teleoperation`. Su objetivo es conservar
el contexto de las decisiones de implementacion y de las calibraciones que son
especificas de la maquina.

## Pose inicial y activacion

- Se conservaron tres poses: `down`, `arms_90` y `forward_open`.
- `forward_open` es la predeterminada: abre simetricamente los codos con J2,
  recentra las manos con J3 y mantiene J4 a 90 grados apuntando hacia delante.
- El backend real lee y estabiliza primero las posiciones articulares medidas.
- Para evitar la columna central, el movimiento real hacia `forward_open`
  primero establece lentamente la apertura de J1-J3 manteniendo J4-J7 en su
  postura medida, y solo despues dobla J4 hacia 90 grados.
- La simulacion acepta `--auto-start` y `--auto-start-delay-s`. Con tracking
  continuo valido inicia una cuenta regresiva de cinco segundos y usa el mismo
  camino seguro de activacion que Space.
- La cuenta regresiva se cancela si se pierde el tracking.

## Backend real y seguridad

- J8 se registra en modo `POS_FORCE`, como requiere la API de posicion del
  gripper de OpenArm.
- Se descartan las primeras muestras frias de CAN antes de adoptar la pose
  inicial medida.
- El watchdog no cancela el objetivo mientras se ejecuta el movimiento
  bloqueante hacia home.
- Los errores de home y following error indican brazo, joint, posicion medida
  y objetivo para facilitar el diagnostico.
- El gripper inicia cerrado y despues recibe las aperturas medidas por HandUMI.
- La apertura normalizada `[0, 1]` se convierte a los endpoints fisicos de J8
  calibrados para cada brazo.
- Una diferencia numerica diminuta en el limite URDF se ajusta al limite.
- Si una solucion IK sale realmente del rango, se rechaza el objetivo completo
  de ese brazo y se conserva su ultima consigna segura. No se recortan joints
  individuales porque eso modifica la postura cartesiana. El otro brazo y los
  grippers pueden continuar, y el brazo retenido se recupera cuando el IK
  regresa al rango.
- Un following error fisico sigue deteniendo el streamer: puede indicar atasco,
  motor desconectado o una discrepancia peligrosa y no debe ocultarse.

## Setup CAN y zero position

- El wizard de OpenArm detecta las interfaces CAN disponibles y pregunta cual
  corresponde al brazo derecho y al izquierdo, sin reutilizar el flujo de
  desconexion de Piper.
- Rechaza asignar la misma interfaz a ambos brazos.
- Configura CAN-FD a 1 Mbps nominal y 5 Mbps de datos.
- Exige respuesta de J1-J8 en cada brazo y da un diagnostico especifico cuando
  falta un motor o su baudrate interno no es 5 Mbps.
- La calibracion oficial de zero position se ejecuta un brazo a la vez mediante
  `--openarm-zero-side right|left|both` y exige una confirmacion explicita por
  lado.
- El calibrador se ejecuta dentro del entorno Python activo y no se sustituye
  por `set_zero`.

## Calibracion de grippers

- Se agrego `handumi-calibrate-openarm-grippers` para capturar por separado los
  endpoints cerrado y abierto de J8.
- El modo predeterminado es manual: J8 queda deshabilitado y el operador mueve
  el mecanismo sin torque.
- La captura verifica estabilidad antes de aceptar una posicion.
- Existe un modo automatico experimental basado en torque, pero no es el flujo
  recomendado.
- Los resultados se guardan en
  `~/.cache/handumi/openarmv1_grippers.yaml` y se cargan automaticamente en
  teleoperacion real.

## Controller a TCP

- Se elimino el duplicado `openarmv1_pico_controller_tcp.yaml`.
- Todo robot que use PICO resuelve la calibracion compartida
  `configs/calibration/pico_controller_tcp.yaml`; sus metadatos de captura se
  conservaron en ese unico archivo.
- PICO y Meta mantienen calibraciones independientes. Controller-a-TCP tampoco
  es la calibracion mecanica de J8 ni la calibracion Feetech.

## Calibraciones locales actuales

Las siguientes mediciones son locales y no se versionan:

```yaml
# ~/.cache/handumi/calibration.yaml
left:  {closed_ticks: 1367, open_ticks: 2366, max_width_mm: 90.0}
right: {closed_ticks: 2433, open_ticks: 1459, max_width_mm: 90.0}

# ~/.cache/handumi/openarmv1_grippers.yaml
right: {closed_position_rad: 0.9645609216449227, open_position_rad: -0.1958876935988414}
left:  {closed_position_rad: -0.00705729762722207, open_position_rad: -1.1938277256427856}
```

## Video hacia PICO

- El primer experimento basado en un sender generico de webcam fue eliminado
  por completo y no forma parte del repositorio.
- La integracion correcta usa XRoboToolkit Remote Vision: canal de comandos en
  `13579`, stream H.264 en `12345`, `adb reverse` para comandos y `adb forward`
  para video.
- La fuente seleccionada dentro del visor es `ZEDMINI`; no se reemplaza el
  `video_source.yml` instalado en el PICO.
- La captura real se adapta al protocolo usado por
  `wbcd-icra-2026-deployment/gear_sonic/scripts/pico_sim_vision_bridge.py`.
- El bridge queda como comando experimental standalone para cualquier robot;
  no agrega flags ni ciclo de vida de camara a `handumi-teleop-real`.
- Se corrigio el empaquetado H.264 para conservar los NAL con start code de
  tres bytes dentro de la unidad Annex-B de cuatro bytes que espera el decoder
  de XRoboToolkit.

## Verificacion acumulada

- La suite completa despues de los cambios de OpenArm termino con `280 passed`.
- `git diff --check` no reporto errores de whitespace.
