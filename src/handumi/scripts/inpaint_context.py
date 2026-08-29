#!/usr/bin/env python3
"""Repaint a dataset's context camera so it shows the target robot, not the operator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv

from handumi.inpainting import (
    DEFAULT_MAX_CALLS,
    WORKSPACE_VIDEO_KEY,
    ClipSpec,
    Ledger,
    align_to_source,
    composite,
    edit_clip,
    edit_mask,
    evaluate,
    extract_clip,
    extract_reference,
    file_sha256,
    load_prompt,
    now_iso,
    prompt_path,
    read_video,
    resolve_episode_clip,
    write_inpainted_dataset,
)
from handumi.inpainting.omni import API_KEY_ENV, DEFAULT_RESOLUTION, MODEL
from handumi.robots.registry import EMBODIMENT_NAMES

load_dotenv()


def parse_episodes(value: str) -> list[int]:
    """Accept ``3``, ``0,2,5`` and ``0-9,12`` alike."""
    episodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            start, _, end = part.partition("-")
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    if not episodes:
        raise argparse.ArgumentTypeError(f"No episodes in {value!r}.")
    return sorted(dict.fromkeys(episodes))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Edit a dataset's context-camera footage with Gemini Omni Flash so the "
            "operator's arm is replaced by the target robot, verify each episode "
            "against its recorded frames, and optionally write the derivative dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", type=Path, help="Local dataset root.")
    parser.add_argument("--robot", choices=EMBODIMENT_NAMES, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--episodes",
        type=parse_episodes,
        help="Episodes to process: 3, or 0,2,5, or 0-9,12.",
    )
    selection.add_argument(
        "--all", action="store_true", help="Every episode in the dataset."
    )
    parser.add_argument(
        "--first-frame",
        type=int,
        default=0,
        help="Offset within each episode, not within the video file.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Cap the clip length. Defaults to the whole episode; an episode over "
        "the upload limit is time-compressed for the call and restored on return.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        required=True,
        help="Robot reference image or clip; repeat for more (bound as <IMAGE_REF_0>, ...).",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help="Defaults to configs/inpainting-prompts; the prompt is <robot>.md.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/inpainting/<dataset>.",
    )
    parser.add_argument(
        "--write-dataset",
        type=Path,
        default=None,
        help="After generating, write the derivative dataset here.",
    )
    parser.add_argument(
        "--allow-partial-episode",
        action="store_true",
        help="Proceed even though a clip does not span its episode. The video will "
        "be shorter than that episode's action rows.",
    )
    parser.add_argument(
        "--anchor-mask",
        action="store_true",
        help="Bound the edit to where the operator was: better scene preservation, "
        "more flicker along the mask edge.",
    )
    parser.add_argument(
        "--resolution",
        choices=("360p", "720p", "1080p", "4k"),
        default=DEFAULT_RESOLUTION,
        help="Generated video resolution. Output is billed per second by resolution, "
        "and anything above the camera's own frame size is discarded by the resample.",
    )
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Actually spend generation calls. Without it this is a dry run.",
    )
    return parser


def _encode(frames, path: Path, fps: int) -> Path:
    """Write frames losslessly, then transcode to H.264 for playback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    lossless = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(lossless), cv2.VideoWriter.fourcc(*"FFV1"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(lossless),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path)],
        check=True,
    )
    lossless.unlink()
    return path


def _side_by_side(source: Path, result: Path, output: Path, labels: tuple[str, str]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    top, bottom = labels
    style = "fontsize=16:fontcolor=white:box=1:boxcolor=black@0.6:x=8:y=8"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-i", str(result),
         "-filter_complex",
         f"[0:v]drawtext=text='{top}':{style}[a];"
         f"[1:v]drawtext=text='{bottom}':{style}[b];[a][b]vstack",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(output)],
        check=True,
    )
    return output


def _prepare(args, episode: int, run_dir: Path) -> tuple[Path, Path, ClipSpec]:
    """Cut the upload clip and the native-rate reference for one episode.

    They differ whenever the episode is long enough to be time-compressed for the
    call: the upload is 10s, the reference keeps the episode's own frames, and
    compositing and grading run against the reference.
    """
    source_video, spec = resolve_episode_clip(
        args.dataset,
        episode,
        video_key=WORKSPACE_VIDEO_KEY,
        first_frame=args.first_frame,
        max_frames=args.frames,
    )
    if not source_video.exists():
        raise SystemExit(f"Context-camera video not found: {source_video}")
    if not spec.covers_episode and not args.allow_partial_episode:
        raise SystemExit(
            f"Episode {episode} is {spec.episode_frames} frames and the clip covers "
            f"{spec.first_frame + spec.frames}; {spec.missing_frames} frames "
            f"({spec.missing_frames / spec.fps:.2f}s) would have action rows and no picture.\n"
            f"Raise --frames to span the episode, or pass --allow-partial-episode."
        )
    clip_path = run_dir / "input" / f"ep{episode:03d}.mp4"
    if not clip_path.exists():
        extract_clip(source_video, spec, clip_path)
    reference = run_dir / "input" / f"ep{episode:03d}_reference.mp4"
    if not reference.exists():
        if spec.time_scale == 1.0:
            reference = clip_path
        else:
            extract_reference(source_video, spec, reference)
    return clip_path, reference, spec


def _process(args, episode: int, run_dir: Path, prompt: str, references: list[Path],
             ledger: Ledger) -> dict[str, Any]:
    """Generate, align, composite and grade one episode. Spends one call."""
    clip_path, reference, spec = _prepare(args, episode, run_dir)
    scale = f", {spec.time_scale:.3f}x time-compressed" if spec.time_scale > 1 else ""
    print(f"  ep{episode:03d}: {spec.frames} frames, {spec.seconds:.3f}s{scale}")

    timestamp = now_iso()
    intent = {
        "run_id": f"{timestamp}#{episode}",
        "episode": episode,
        "timestamp": timestamp,
        "phase": "intent",
        "spent_call": True,
        "robot": args.robot,
        "clip": clip_path.name,
        "resolution": args.resolution,
        "anchored_mask": bool(args.anchor_mask),
        "episode_frames": spec.episode_frames,
        "clip_frames": spec.frames,
        "time_scale": round(spec.time_scale, 4),
        "covers_episode": spec.covers_episode,
        "prompt_file": str(prompt_path(args.robot, args.prompts_dir)),
        "prompt": prompt,
        "references": [{"path": str(p), "sha256": file_sha256(p)} for p in references],
    }
    ledger.append(intent)  # logged before the call, so a crash cannot lose the count

    raw = run_dir / "raw" / f"ep{episode:03d}_raw.mp4"
    try:
        result = edit_clip(clip_path, references, prompt, raw, resolution=args.resolution)
    except Exception as exc:
        # A call the API rejected generated no video and was never billed.
        ledger.append({
            "run_id": intent["run_id"], "episode": episode, "timestamp": now_iso(),
            "phase": "refused", "spent_call": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise

    aligned = align_to_source(raw, spec, run_dir / "raw" / f"ep{episode:03d}_aligned.mp4")
    source = read_video(reference)
    generated = read_video(aligned)
    composited = composite(
        source, generated, edit_mask(source, generated, anchored=args.anchor_mask)
    )
    context_video = _encode(composited, run_dir / "context" / f"ep{episode:03d}.mp4", spec.fps)
    side_by_side = _side_by_side(
        reference, context_video,
        run_dir / "review" / f"ep{episode:03d}_side_by_side.mp4",
        ("INPUT - operator", f"EPISODE {episode} - {args.robot}"),
    )

    report = evaluate(
        source, generated, read_video(context_video),
        episode=episode, frames_expected=spec.frames,
    )
    report.write(run_dir / "metrics" / f"ep{episode:03d}.json")
    ledger.append({
        **intent, "phase": "result", "timestamp": now_iso(),
        "interaction_id": result.interaction_id, "status": result.status,
        "latency_s": result.latency_s, "gates": report.to_dict(),
        "artifacts": {"context_video": str(context_video), "raw": str(raw),
                      "side_by_side": str(side_by_side)},
    })

    flags = " ".join(f"[{f.severity}:{f.code}]" for f in report.findings)
    print(f"    {report.frames_got}/{spec.frames} frames, {result.latency_s}s, "
          f"{report.to_dict()['status']} {flags}".rstrip())
    return report.to_dict()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    info_path = args.dataset / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"Not a dataset root: {args.dataset}")
    total = int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])
    episodes = list(range(total)) if args.all else args.episodes

    run_dir = args.output_dir or Path(f"outputs/inpainting/{args.dataset.name}")
    ledger = Ledger(run_dir / "ledger.jsonl")
    budget = ledger.budget(args.max_calls)
    print(f"dataset {args.dataset} | {len(episodes)} episode(s) | api calls used: {budget}")

    try:
        prompt = load_prompt(args.robot, args.prompts_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    references = [p.resolve() for p in args.reference]
    for reference in references:
        if not reference.exists():
            raise SystemExit(f"Reference not found: {reference}")

    if not args.commit:
        for episode in episodes:
            *_, spec = _prepare(args, episode, run_dir)
            scale = f"  ({spec.time_scale:.3f}x compressed)" if spec.time_scale > 1 else ""
            print(f"  ep{episode:03d}: {spec.frames} frames, {spec.seconds:.3f}s{scale}")
        print(f"\nDRY RUN - would spend {len(episodes)} call(s) at {args.resolution}. "
              f"Re-run with --commit.")
        return

    if not os.environ.get(API_KEY_ENV):
        raise SystemExit(f"{API_KEY_ENV} is not set (put it in .env).")

    reports: list[dict[str, Any]] = []
    for episode in episodes:
        budget = ledger.budget(args.max_calls)
        if budget.exhausted:
            print(f"\nBudget exhausted at {budget}; stopping before ep{episode:03d}. "
                  f"Raise --max-calls deliberately to continue.")
            break
        try:
            reports.append(_process(args, episode, run_dir, prompt, references, ledger))
        except Exception as exc:  # keep the run going; the ledger holds the reason
            print(f"    ep{episode:03d} FAILED: {type(exc).__name__}: {exc}")

    summary = {
        "dataset": str(args.dataset), "robot": args.robot, "resolution": args.resolution,
        "generated_at": now_iso(), "model": MODEL,
        "api_calls_used": ledger.budget(args.max_calls).used,
        "episodes": reports,
        "accepted": [r["episode"] for r in reports if r["status"] == "accepted"],
        "rejected": [r["episode"] for r in reports if r["status"] != "accepted"],
    }
    (run_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{len(summary['accepted'])} accepted, {len(summary['rejected'])} rejected "
          f"| calls used {summary['api_calls_used']}/{args.max_calls}")
    print(f"report: {run_dir / 'report.json'}")

    if args.write_dataset:
        videos = {
            r["episode"]: run_dir / "context" / f"ep{r['episode']:03d}.mp4"
            for r in reports if r["status"] == "accepted"
        }
        if not videos:
            raise SystemExit("No accepted episodes to write.")
        written = write_inpainted_dataset(
            args.dataset, args.write_dataset, videos,
            provenance={
                "model": MODEL, "robot": args.robot, "resolution": args.resolution,
                "prompt_file": str(prompt_path(args.robot, args.prompts_dir)),
                "prompt_sha256": file_sha256(prompt_path(args.robot, args.prompts_dir)),
                "references": [
                    {"path": str(p), "sha256": file_sha256(p)} for p in references
                ],
            },
        )
        print(f"\ndataset: {written.output}  "
              f"({len(written.inpainted_episodes)} inpainted, "
              f"{len(written.passthrough_episodes)} original, "
              f"{written.frames_written} frames)")


if __name__ == "__main__":
    sys.exit(main())
