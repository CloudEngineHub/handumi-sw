"""Probe OpenCV camera indices for HandUMI wrist-camera assignment."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required to probe cameras.") from exc

    found: list[int] = []
    for index in range(args.start_index, args.end_index + 1):
        cap = cv2.VideoCapture(index)
        ok = cap.isOpened()
        if ok:
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"{index}: OK {w}x{h}")
                found.append(index)
            else:
                print(f"{index}: opened but no frame")
        cap.release()

    if not found:
        print("No cameras found.")


if __name__ == "__main__":
    main()
