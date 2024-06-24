import os
import time
import builtins
import uuid
import pathlib
import shutil

original_open = builtins.open

def safe_open(filename, mode='r', **kwargs):
    if 'w' in mode and os.path.isfile(filename) and shutil.disk_usage('/').free > 2**34:
        p = pathlib.Path(filename)
        suffix = f"{os.path.getctime(filename)}_{time.time()}_{uuid.uuid4().hex}"
        newname = str(p.with_stem(f"{p.stem}_{suffix}"))
        os.rename(filename, newname)
    else:
        newname = filename
    return original_open(filename, mode=mode, **kwargs)

builtins.open = safe_open
