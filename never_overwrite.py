import os
import datetime
import builtins
import uuid
import pathlib

original_open = builtins.open

def safe_open(filename, mode='r', **kwargs):
    if 'w' in mode and os.path.isfile(filename):
        p = pathlib.Path(filename)
        suffix = str(os.path.getctime(filename)) + datetime.datetime.utcnow().strftime('_%Y_%m_%d_%H_%M_%S.%f_') + uuid.uuid4().hex
        newname = str(p.with_stem(f"{p.stem}_{suffix}"))
        os.rename(filename, newname)
    else:
        newname = filename
    return original_open(filename, mode=mode, **kwargs)

builtins.open = safe_open
