# info.py

import sys

def info(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
