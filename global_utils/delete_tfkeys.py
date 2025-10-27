import os
import re
import fnmatch
import shutil
import unicodedata
from typing import Iterable, List, Tuple, Callable, Set, Dict

from tensorboardX.proto import event_pb2
import google_crc32c
import struct

# ---------------- TFRecord framing ----------------

def _masked_crc32c(data: bytes) -> int:
    crc = google_crc32c.value(data)
    masked = ((crc >> 15) | ((crc & 0xFFFFFFFF) << 17)) & 0xFFFFFFFF
    masked = (masked + 0xA282EAD8) & 0xFFFFFFFF
    return masked

def _read_tfrecords(fp) -> Iterable[bytes]:
    u64 = struct.Struct("<Q")
    u32 = struct.Struct("<I")
    while True:
        hdr = fp.read(u64.size)
        if not hdr:
            return
        if len(hdr) != u64.size:
            raise IOError("Corrupt TFRecord: short length header")
        (length,) = u64.unpack(hdr)

        len_crc_bytes = fp.read(u32.size)
        if len(len_crc_bytes) != u32.size:
            raise IOError("Corrupt TFRecord: short len CRC")
        (len_crc_read,) = u32.unpack(len_crc_bytes)
        if len_crc_read != _masked_crc32c(hdr):
            raise IOError("Corrupt TFRecord: length CRC mismatch")

        data = fp.read(length)
        if len(data) != length:
            raise IOError("Corrupt TFRecord: short data")

        data_crc_bytes = fp.read(u32.size)
        if len(data_crc_bytes) != u32.size:
            raise IOError("Corrupt TFRecord: short data CRC")
        (data_crc_read,) = u32.unpack(data_crc_bytes)
        if data_crc_read != _masked_crc32c(data):
            raise IOError("Corrupt TFRecord: data CRC mismatch")

        yield data

def _write_tfrecord_record(fp, data: bytes) -> None:
    u64 = struct.Struct("<Q")
    u32 = struct.Struct("<I")
    length = len(data)
    len_hdr = u64.pack(length)
    fp.write(len_hdr)
    fp.write(u32.pack(_masked_crc32c(len_hdr)))
    fp.write(data)
    fp.write(u32.pack(_masked_crc32c(data)))

# ---------------- Event helpers ----------------

def _iter_event_messages(event_path: str) -> Iterable[event_pb2.Event]:
    with open(event_path, "rb") as f:
        for rec in _read_tfrecords(f):
            ev = event_pb2.Event()
            ev.ParseFromString(rec)
            yield ev

def _has_non_summary_payload(ev: event_pb2.Event) -> bool:
    return any([
        bool(ev.file_version),
        ev.HasField("graph_def"),
        ev.HasField("meta_graph_def"),
        ev.HasField("session_log"),
        ev.HasField("tagged_run_metadata"),
    ])

def _normalize_tag(tag: str) -> str:
    # Strip and Unicode-normalize so visually same strings compare equal
    return unicodedata.normalize("NFC", tag).strip()

def _build_matcher(keys: List[str], mode: str) -> Callable[[str], bool]:
    """
    Returns a predicate(tag) -> bool according to match mode.
    """
    norm_keys = [ _normalize_tag(k) for k in keys ]
    if mode == "exact":
        keyset = set(norm_keys)
        return lambda tag: _normalize_tag(tag) in keyset
    elif mode == "prefix":
        return lambda tag: any(_normalize_tag(tag).startswith(k) for k in norm_keys)
    elif mode == "regex":
        regs = [ re.compile(k) for k in norm_keys ]
        return lambda tag: any(r.search(_normalize_tag(tag)) for r in regs)
    elif mode == "glob":
        return lambda tag: any(fnmatch.fnmatchcase(_normalize_tag(tag), patt) for patt in norm_keys)
    else:
        raise ValueError(f"Unknown match mode: {mode}")

def _find_event_files(root_dir: str, max_depth: int = 3) -> List[str]:
    event_files = []
    root_dir = os.path.abspath(root_dir)
    for cur_root, dirs, files in os.walk(root_dir):
        rel = os.path.relpath(cur_root, root_dir)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue
        for f in files:
            if "tfevents" in f:
                event_files.append(os.path.join(cur_root, f))
    return event_files

# ---------------- Diagnostics ----------------

def dump_all_tags(root_dir: str, max_depth: int = 3) -> Dict[str, Set[str]]:
    """
    Scan all event files and return {file_path: set(tags)}.
    Also prints a quick summary.
    """
    files = _find_event_files(root_dir, max_depth)
    if not files:
        print("No TensorBoard event files found.")
        return {}

    result: Dict[str, Set[str]] = {}
    total = 0
    for path in files:
        tags: Set[str] = set()
        try:
            for ev in _iter_event_messages(path):
                if ev.HasField("summary"):
                    for v in ev.summary.value:
                        tags.add(_normalize_tag(v.tag))
        except Exception as ex:
            print(f"⚠️  Error reading {path}: {ex}")
            continue
        result[path] = tags
        total += len(tags)
        print(f"{path}: {len(tags)} tags")
    print(f"\nUnique tag inventory across {len(files)} files collected. Total tag entries: {total}")
    return result

# ---------------- Main cleaner ----------------

def clean_tensorboard_logs(
    root_dir: str,
    keys_to_remove: List[str],
    dry: bool = True,
    max_depth: int = 3,
    match_mode: str = "exact",   # "exact" | "prefix" | "regex" | "glob"
    show_near_misses: int = 8,   # print closest matches when none found
) -> List[Tuple[str, List[str]]]:
    """
    Remove Summary tags from TensorBoard event files (no TensorFlow required).

    Returns list of (event_file_path, sorted(list_of_tags_matched)).
    """
    matcher = _build_matcher(keys_to_remove, match_mode)
    event_files = _find_event_files(root_dir, max_depth)
    if not event_files:
        print("No TensorBoard event files found.")
        return []

    # Build quick index of tags per file for diagnostics
    tag_inventory = {}
    for path in event_files:
        try:
            tags = set()
            for ev in _iter_event_messages(path):
                if ev.HasField("summary"):
                    for v in ev.summary.value:
                        tags.add(_normalize_tag(v.tag))
            tag_inventory[path] = tags
        except Exception as ex:
            print(f"⚠️  Error scanning {path}: {ex}")

    changes_summary: List[Tuple[str, List[str]]] = []

    for path in event_files:
        print(f"\nProcessing: {path}")
        found_tags = set()
        filtered_serialized: List[bytes] = []

        try:
            for ev in _iter_event_messages(path):
                if not ev.HasField("summary"):
                    filtered_serialized.append(ev.SerializeToString())
                    continue

                keep_vals = []
                for val in ev.summary.value:
                    tag_norm = _normalize_tag(val.tag)
                    if matcher(tag_norm):
                        found_tags.add(tag_norm)
                    else:
                        keep_vals.append(val)

                if keep_vals:
                    new_ev = event_pb2.Event()
                    new_ev.CopyFrom(ev)
                    del new_ev.summary.value[:]
                    new_ev.summary.value.extend(keep_vals)
                    filtered_serialized.append(new_ev.SerializeToString())
                else:
                    if _has_non_summary_payload(ev):
                        filtered_serialized.append(ev.SerializeToString())

            if not found_tags:
                print("  - No matching tags found with mode="
                      f"{match_mode}.")
                # Help user by showing some nearby tags (same directory-like prefix)
                if show_near_misses and path in tag_inventory:
                    inv = sorted(tag_inventory[path])
                    # heuristic: show tags that share the longest common prefix with any key
                    def lcp(a, b):
                        m = 0
                        for x, y in zip(a, b):
                            if x == y:
                                m += 1
                            else:
                                break
                        return m
                    candidates = []
                    for key in keys_to_remove:
                        keyn = _normalize_tag(key)
                        scored = sorted(inv, key=lambda t: -lcp(t, keyn))
                        candidates.extend(scored[:max(2, show_near_misses//len(keys_to_remove) or 1)])
                    # dedup but keep order
                    seen = set()
                    hints = [t for t in candidates if not (t in seen or seen.add(t))]
                    print("  ~ Nearby tags in this file:")
                    for h in hints[:show_near_misses]:
                        print(f"    · {h}")
                changes_summary.append((path, []))
                continue

            print("  - Tags matched:", ", ".join(sorted(found_tags)))
            changes_summary.append((path, sorted(found_tags)))

            if dry:
                print("  (dry run) Would remove listed tags from this file.")
                continue

            # Real rewrite
            backup_path = path + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(path, backup_path)

            tmp_path = path + ".cleaning.tmp"
            with open(tmp_path, "wb") as out:
                for rec in filtered_serialized:
                    _write_tfrecord_record(out, rec)
            os.replace(tmp_path, path)
            print("  ✓ Cleaned file written. Original saved as .bak")
        except Exception as ex:
            print(f"  ⚠️ Error processing {path}: {ex}")

    return changes_summary


default_keys = ",".join(["validation/train_loss/dataloader_idx_1",
                         "validation/train_dice/dataloader_idx_1",
                         "validation/val_dice/dataloader_idx_0",
                         "validation/val_loss/dataloader_idx_0",
                         "hp_metric"])

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean TensorBoard logs by removing specified keys.")
    parser.add_argument("--root_dir", type=str, default="/home/jloch/Desktop/diff/luzern/values/saves/lidc-small-snn/version_2", 
                        help="Root directory to search for TensorBoard event files.")
    parser.add_argument("--keys", type=str, help="List of TensorBoard scalar keys to remove.", default=default_keys)
    parser.add_argument("--dry", action='store_true', help="If set, only prints which keys would be deleted without making changes.")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum search depth relative to root_dir.")

    args = parser.parse_args()
    clean_tensorboard_logs(args.root_dir, args.keys.split(','), dry=args.dry, max_depth=args.max_depth)
    print(dump_all_tags(args.root_dir, max_depth=3))
if __name__ == "__main__":
    main()