"""录一段无损双目数据（阶段一：压缩伪影量化的输入）。

    python -m pi.tools.record_pairs --out data/recordings/calib_scene --frames 300

无相机时 backend=synthetic，同样会写出标准 pairs.jsonl + PNG。
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from pi.capture.factory import build_backend
from pi.stream.publisher import FilePublisher
from shared.config import load_config
from shared.recording import write_json


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="录制无损双目帧对")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--backend", default="auto", choices=["auto", "picamera2", "synthetic"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jpeg", action="store_true", help="默认 PNG 无损；加此项存 JPEG")
    args = ap.parse_args(argv)

    cfg = load_config(None, {"camera.backend": args.backend, "camera.fps": args.fps})
    backend = build_backend(cfg)
    pub = FilePublisher(args.out, image_format="jpg" if args.jpeg else "png")
    backend.start()
    n = 0
    try:
        for _ in range(args.frames):
            pair = backend.read()
            pub.publish(pair)
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{args.frames}", flush=True)
    finally:
        backend.close()
        pub.close()

    write_json(
        f"{args.out}/capture.json",
        {"frames": n, "backend": args.backend, "format": "jpg" if args.jpeg else "png"},
    )
    print(f"完成：{n} 对 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
