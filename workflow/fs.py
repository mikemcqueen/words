import errno

from pathlib import Path

def not_a_directory_error(path: Path): 
    return NotADirectoryError(errno.ENOTDIR, "not a directory:", path)


def not_a_file_error(path: Path): 
    return ValueError(f"not a regular file: {path}")


def file_not_found_error(path: Path): 
    return FileNotFoundError(errno.ENOENT, "file not found:", path)


def file_already_exists_error(path: Path): 
    return ValueError(f"file already exists: {path}")


def raise_if_not_exists(path: Path):
    if not path.exists():
        raise file_not_found_error(path)


def raise_if_exists(path: Path):
    if path.exists():
        raise file_already_exists_error(path)


def raise_if_not_dir(path: Path):
    raise_if_not_exists(path)
    if not path.is_dir():
        raise not_a_directory_error(path)


def raise_if_not_file(path: Path):
    raise_if_not_exists(path)
    if not path.is_file():
        raise not_a_file_error(path)
    
