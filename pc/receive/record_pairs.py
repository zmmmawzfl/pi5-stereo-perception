"""第一天 PC 端最小接收器：按 pair_id 落盘，而不是"先左后右"。

用法::

    python -m pc.receive.record_pairs --endpoint tcp://192.168.1.10:5555 \
        --out data/recordings/day1 --max-pairs 100

输出 summary.json：接收/配对/重复/孤儿/乱序计数，以及时间戳差分布。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from pc.receive.subscriber import ZmqPairSource
from shared.recording import PairRecorder, write_json


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PC 端最小接收器（按 pair_id 配对落盘）")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    ap.add_argument("--out", required=True, help="录制目录")
    ap.add_argument("--max-pairs", type=int, default=100)
    ap.add_argument("--idle-timeout", type=float, default=15.0, help="无帧超时秒数")
    ap.add_argument("--stale", type=float, default=1.0, help="单边帧超时秒数")
    ap.add_argument("--format", default="png", choices=["png", "jpg"])
    args = ap.parse_args(argv)

    source = ZmqPairSource(args.endpoint, stale_s=args.stale)
    root = Path(args.out)
    got = 0
    with PairRecorder(root, image_format=args.format) as rec:
        try:
            for rp in source.pairs(max_pairs=args.max_pairs, idle_timeout_s=args.idle_timeout):
                rec.add(
                    rp.left,
                    rp.right,
                    rp.pair.left.sensor_ts_ns,
                    rp.pair.right.sensor_ts_ns,
                    rp.pair_id,
                )
                got += 1
                if got % 10 == 0:
                    print(f"[recv] {got} 对，最近 skew={rp.ts_diff_ns / 1e6:.3f} ms", flush=True)
        except KeyboardInterrupt:
            print("\n[recv] 中断", flush=True)

    summary = {
        "endpoint": args.endpoint,
        "recorded_pairs": got,
        "pairing": source.stats.as_dict(),
        "output": str(root),
    }
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if got == 0:
        print("[recv] 没有收到成对的帧：检查 Pi 是否在推流、端点/防火墙、SUB/SUB 端口", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
