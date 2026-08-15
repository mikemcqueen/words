import sys

CODES = {
    "green":        "\x1b[32m",
    "bright green": "\x1b[92m",
    "white":        "\x1b[37m",
    "yellow":       "\x1b[33m",
    "red":          "\x1b[31m",
    "off":          "\x1b[0m"
}

def _msg(color: str, msg: str, file):
    print(f'{CODES[color]}{msg}{CODES["off"]}', file=file)


def success(msg: str, file=None):
    _msg("bright green", msg, file=sys.stdout if file is None else file)


def info(msg: str, file=None):
    _msg("white", msg, file=sys.stderr if file is None else file)


def warn(msg: str, file=None):
    _msg("yellow", msg, file=sys.stderr if file is None else file)


def error(msg: str, file=None):
    _msg("red", msg, file=sys.stderr if file is None else file)
