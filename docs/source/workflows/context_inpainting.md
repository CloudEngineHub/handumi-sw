# Repaint the Context Camera

HandUMI's wrist cameras are embodiment-agnostic by construction: they see the
gripper fins and the scene, and nothing that identifies who is holding the tool.
The context camera is not. It sees the operator's forearm, hand, the plastic
shells, the controller and its cables, so a policy trained on that stream learns
a world in which a human arm reaches into the scene, and then meets a robot at
deployment.

`handumi dataset inpaint` rewrites one context-camera clip so it shows the
target robot instead, using Gemini Omni Flash to edit the recorded video.

:::{warning}
Generation is billed per call. The command edits **one clip of one episode** and
refuses to run past its call budget. There is no batch mode: scaling to a whole
dataset is a separate decision, taken deliberately.
:::

## Prerequisites

```bash
uv sync --extra inpaint
echo 'GEMINI_API_KEY="..."' >> .env
```

The key needs a project with billing enabled; the free tier's quota for video
generation is zero.

## Run it

```bash
handumi dataset inpaint outputs/datasets/handumi-demo \
  --robot piper \
  --episodes 0 \
  --reference assets/piper/piper-photo.jpeg
```

`--episodes` takes `3`, `0,2,5` or `0-9,12`; `--all` takes the whole dataset. The
call budget is shared across the run and checked before each episode, so a batch
stops at its limit instead of running past it. `--write-dataset PATH` writes the
derivative dataset when the run finishes.

That is a **dry run**: it extracts the clip, resolves the prompt and references,
prints exactly what would be sent, and spends nothing. Add `--commit` to issue
the call.

Everything lands under `outputs/inpainting/<dataset>_ep<NNN>/`:

```text
context/ep000.mp4               THE DELIVERABLE: the new context camera
input/ep000.mp4                 the extracted source clip
raw/ep000_raw.mp4               what the model returned, untouched
raw/ep000_aligned.mp4           resampled onto the dataset frame grid
review/ep000_side_by_side.mp4   input above, result below
metrics/ep000.json              that episode's gates
report.json                     every episode, accepted and rejected
ledger.jsonl                    every intent, result and gate, appended
```

One run and one video per episode. Re-running an episode replaces its artifacts,
so the directory never fills with candidates to choose between; the ledger keeps
the history of what produced the one on disk, including the prompt and the
reference hashes.

The bytes the model returns are written to `raw/` before anything else touches
them: the call was already paid for by the time they arrived.

## Cost

Video output dominates the bill and is priced per second by resolution, so the
default is **360p**: the context camera is 672x376, and anything larger is paid
for and then discarded by the resample back onto the dataset frame grid. At 720p
a ten-second clip costs roughly three times as much for pixels that never
survive.

To try a prompt change cheaply, shorten the clip rather than the resolution:
`--frames 90` is three seconds, and enough to see whether the arm has the right
colour and whether the operator is gone.

## Long episodes

The editing API caps the **duration** it accepts, not the frame count: ten
seconds per upload. In a forty-episode session measured here, ten exceeded it,
the longest at 15.57 s.

They are not cut short. An episode over the cap is time-compressed for the call
and expanded again on return, so one call still covers the whole episode and the
output lands on the dataset's frame grid exactly. Episode 29 goes out at 1.557x
and comes back as its original 467 frames.

What that costs is the *temporal density of the edit*, not frames. The model
edits roughly 240 distinct frames however long the episode is, so on a 15-second
episode the arm is effectively edited at 15 fps and interpolated onto a 30 fps
background. Note this is a difference of degree, not of kind: the model already
answers at 24 fps for a clip that fits, and `align_to_source` already stretches
that to 30.

`--frames` caps the clip when you want a sub-range — `--frames 90` is three
seconds, useful for trying a prompt change cheaply. A clip that does not span
its episode is refused unless you pass `--allow-partial-episode`: frames with an
action row and no picture would mislabel the dataset.

## Writing the dataset

The per-episode videos are not a dataset. `--write-dataset PATH` copies the
source and rebuilds `observation.images.workspace` from them, leaving every other
stream, every action row and every timestamp untouched.

Episodes sit back to back inside each stream file, so the rewritten stream is
one segment per episode in order: the inpainted clip where there is one, the
recorded frames where there is not. A partially inpainted dataset is therefore
still complete and playable rather than one with holes.

Two refusals protect it. A clip whose frame count differs from its episode's row
count is refused outright, and so is a rewrite whose total frames do not match
the rows. The metadata records the model, the prompt and its hash, the reference
hashes, and which episodes were repainted.

## Recovering the camera pose

`handumi calibrate workspace-from-video DATASET` solves the context camera's
pose in the table frame for a session where the ChArUco stage never ran. It
tracks the operator's controller — bright, compact, and with a known table-frame
position in `observation.state` every frame — and solves PnP against that
trajectory.

On one measured episode it lands 118 inliers from 145 detections with 2.76 px
mean reprojection error, putting the camera 60 cm above the table. It refuses to
write a pose whose error exceeds `--max-error-px`.

That pose is what makes the retarget offset measurable: how far, in pixels, the
retargeted gripper lands from where the operator's was.

## What the gates check

The model returns a re-rendered clip at its own resolution and frame rate, with
an audio track it invented. None of that can reach a dataset unchecked.

| gate | why it exists |
|---|---|
| **frames** | The API guarantees neither frame rate nor frame count. A clip returned at 24 fps and written naively would mislabel every action row. Blocking. |
| **preservation** | Every returned pixel is new, including the ones that had to stay. Compositing restores the recording outside the edit; the gate reports what still moved. |
| **human removal** | Skin fraction, input versus output. Read the reduction, not the absolute value: the detector also fires on warm cloth. |
| **scene cuts** | The model invents cuts unless told not to. One cut mid-episode is a training artefact. |
| **temporal stability** | Frame-to-frame drift in regions the recording holds still. |

Gates decide correctness. Whether the arm actually looks like the robot is a
human judgement, made on the side-by-side.

Findings follow the severity split the rest of the pipeline uses. A **reject** is
mechanical — a frame count that disagrees with the action rows, or a scene cut —
and there is no judgement to add. A **warning** is context: an operator still
faintly visible, scene drift, flicker, or an edit that painted more than usual.
Warnings are reported and do not block, because the reviewer already has to look
at the video anyway. The thresholds sit outside the envelope the calibration
runs produced, so they flag outliers rather than normal variation.

## Choosing references

Reference media teaches the model what the embodiment looks like. Best first:

1. **Real teleoperation footage** from `handumi teleop-record` — the same
   camera, table and lighting. Video references are limited to three clips of
   three seconds each.
2. **Photographs of the arm in the lab.**
3. **A product photograph**, such as `assets/piper/piper-photo.jpeg`. Watch for
   its studio lighting and plain background bleeding into the edit.
4. **A render from the embodiment's MJCF**, when no photograph exists. Render a
   *single* arm: a bimanual render contradicts a prompt asking for one arm, and
   the model follows the picture.

Keep reference media in `outputs/inpainting/references/<robot>/`. It is
gitignored, which is where third-party product imagery belongs.

## Prompts

Prompts live in `configs/inpainting-prompts/<robot>.md`, one per embodiment. The
file is the prompt and is sent verbatim; a file that wants notes alongside it
can put them above a `---` rule or inside HTML comments, and only the
instruction below is sent.

What the first embodiment's runs taught, so the next one's prompt does not
relearn it:

- **Ask for one arm.** A HandUMI episode is performed by a single arm — in
  episode 0 the left controller moves 0.048 m and never opens its gripper while
  the right moves 1.707 m. Asking for two produced a confused edit.
- **Name the whole kinematic chain.** Asking for "a robot arm" produced a short
  floating tube; naming gripper, wrist, forearm, elbow, upper arm and shoulder
  produced an arm that reaches in from the far edge.
- **Name the hand-held rig as something to remove.** Calling it "the gripper"
  protected the very thing that had to go: the shells and the white VR
  controller survived every edit until the prompt named them.
- **Expect a trade.** Each instruction that makes the model paint more also
  makes it preserve less; watch `model_changed_pct` alongside the result.
- **Watch the video, not only the gates.** Bounding the edit to the operator's
  footprint (`--anchor-mask`) halves the pixels the model keeps and scores
  better on preservation, but its mask edge follows the operator and flickers.
  The default is the plain difference mask because that is what looked right on
  playback.

The ledger records which prompt produced which video, so a regression can be
traced back to the wording that caused it.

Write the prompt against the demonstration you actually recorded. A HandUMI
episode is usually performed by **one** arm — in one measured episode the left
controller moves 0.048 m and never opens its gripper while the right moves
1.707 m — so asking for two active arms produces a confused edit.

## Limits worth knowing

- Editing uploaded video is unavailable in the EEA, Switzerland and the UK.
- All generated video carries SynthID watermarking.
- There is no seed: two runs of the same prompt differ. Reproducibility comes
  from the ledger, which records the model, prompt, reference hashes and
  interaction id behind every artifact.
