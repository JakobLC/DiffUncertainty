from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union


def format_checkpoint_subdir(epoch: Optional[Union[int, str]], ema: Optional[bool]) -> Optional[str]:
    """Return the folder name for a given epoch/EMA combination.

    If *epoch* is None or cannot be parsed into an integer, None is returned so
    callers can fall back to the legacy directory layout. The EMA flag is only
    appended when truthy, supporting both bools and string representations that
    were already parsed by Hydra/argparse.
    """
    if epoch is None:
        return None
    try:
        epoch_value = int(epoch)
    except (TypeError, ValueError):
        return None
    tag = f"e{epoch_value}"
    if ema:
        tag += "_ema"
    return tag



"""def infer_epoch_from_path(path: Union[str, Path]) -> Optional[int]:
    #Best-effort epoch extraction from a checkpoint filename.

    #Matches strings such as ``epoch=0050.ckpt`` or ``epoch50_step123.ckpt`` and
    #returns the detected integer. When no epoch information is encoded in the
    #filename, None is returned.
    
    match = re.search(r"epoch(?:=|_)?(\d+)", Path(path).name)
    if match:
        return int(match.group(1))
    return None"""
