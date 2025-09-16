"""
Merge multiple TensorBoard event files in a run directory into a single file,
optionally offsetting steps so the timeline appears continuous.

Usage (example):
    python -m values.utils.merge_tfevents \
        --run_dir /home/jloch/Desktop/diff/luzern/values/saves/lidc-snn/version_6 \
        --consolidated_name events.consolidated

Notes:
 - This reads all event files under run_dir (recursively) and writes a new
   event file at the top-level of run_dir. Default output filename starts
   with 'events.consolidated'.
 - It preserves wall_time and step; you can pass --step_offset to shift all
   steps from earlier files if you want to strictly avoid overlaps.
 - This does not delete or modify existing files; you can archive them after.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def find_event_files(run_dir: Path) -> List[Path]:
    # Match common TB event filenames: events.out.tfevents.*
    files = sorted(run_dir.rglob("events.out.tfevents.*"))
    return [f for f in files if f.is_file()]


def load_accumulators(files: List[Path]) -> List[Tuple[Path, event_accumulator.EventAccumulator]]:
    accs = []
    for f in files:
        acc = event_accumulator.EventAccumulator(str(f))
        acc.Reload()
        accs.append((f, acc))
    return accs


def write_scalar(writer: EventFileWriter, tag: str, step: int, wall_time: float, value: float):
    from tensorboard.compat.proto import summary_pb2
    summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=value)])
    ev = event_pb2.Event(wall_time=wall_time, step=step, summary=summary)
    writer.add_event(ev)


def consolidate(run_dir: Path, out_name: str, step_offset: int = 0, normalize_suffixes: bool = False):
    files = find_event_files(run_dir)
    if not files:
        print(f"No event files found under {run_dir}")
        return

    accs = load_accumulators(files)
    out_path = run_dir / out_name
    writer = EventFileWriter(str(out_path))
    print(f"Writing consolidated events to {out_path}")

    # Collect all scalars from each accumulator and write them sorted by wall_time
    # Optionally normalize '/dataloader_idx_*' suffixes to remove index.
    scalar_events = []
    for f, acc in accs:
        for tag in acc.Tags().get('scalars', []):
            new_tag = tag
            if normalize_suffixes:
                new_tag = re.sub(r"/dataloader_idx_\d+", "", new_tag)
            for ev in acc.Scalars(tag):
                scalar_events.append((ev.wall_time, new_tag, ev.step + step_offset, ev.value))

    # Sort by wall_time to preserve order; then write
    scalar_events.sort(key=lambda x: x[0])
    for wall_time, tag, step, value in scalar_events:
        write_scalar(writer, tag, step, wall_time, value)

    writer.close()
    print("Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Path to the run directory (e.g., .../saves/exp/version_X)")
    p.add_argument("--consolidated_name", default="events.consolidated.tfevents", help="Output event filename")
    p.add_argument("--step_offset", type=int, default=0, help="Optional global step offset to apply")
    p.add_argument("--normalize_suffixes", action="store_true", help="Drop '/dataloader_idx_*' suffixes from tags")
    args = p.parse_args()

    consolidate(Path(args.run_dir), args.consolidated_name, args.step_offset, args.normalize_suffixes)


if __name__ == "__main__":
    main()
