import os
import datetime
import builtins
import uuid

original_open = builtins.open

def safe_open(filename, mode='r', **kwargs):
    if 'w' in mode and os.path.isfile(filename):
        suffix = datetime.datetime.utcnow().strftime('_%Y_%m_%d_%H_%M_%S.%f') + uuid.uuid4().hex
        newname = filename + suffix
    else:
        newname = filename
    return original_open(newname, mode=mode, **kwargs)

builtins.open = safe_open