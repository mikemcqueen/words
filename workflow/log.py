import sys

CODES = {
    "green":        "\x1b[32m",
    "bright green": "\x1b[92m",
    "white":        "\x1b[37m",
    "yellow":       "\x1b[33m",
    "red":          "\x1b[31m",
    "off":          "\x1b[0m"
}

def _msg(color: str, msg: str, file=sys.stdout):
    print(f'{CODES[color]}{msg}{CODES["off"]}', file=file)


def success(msg: str, file=sys.stdout):
    _msg("bright green", msg, file=file)


def info(msg: str, file=sys.stderr):
    _msg("white", msg, file=file)


def warn(msg: str, file=sys.stderr):
    _msg("yellow", msg, file=file)


def error(msg: str, file=sys.stderr):
    _msg("red", msg, file=file)
