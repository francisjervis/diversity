from typing import List, Optional
from pathlib import Path

import tempfile
import gzip
import os
import lzma as xz


def compression_ratio(
        data: List[str],
        algorithm: str = 'gzip',
        verbose: bool = False,
        path: Optional[str] = None
) -> float:
    """ Calculates the compression ratio for a collection of text.
    Args:
        data (List[str]): Strings to compress.
        algorithm (str, optional): Either 'gzip' or 'xz'. Defaults to 'gzip'.
        verbose (bool, optional): Print out the original and compressed size separately. Defaults to False.
        path (str, optional): Path to store temporarily zipped files. Defaults to None.
    Returns:
        float: Compression ratio (original size / compressed size)
    """
    
    temp_dir = None
    if not path:
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name)
    else:
        path = Path(path)

    # Prepare data once
    joined_data = ' '.join(data)
    encoded_data = joined_data.encode('utf-8')

    with (path / 'original.txt').open('wb') as f:
        f.write(encoded_data)

    original_size = (path / "original.txt").stat().st_size

    if algorithm == 'gzip':
        with gzip.GzipFile(str(path / 'compressed.gz'), 'wb') as f:
            f.write(encoded_data)
        compressed_size = (path / "compressed.gz").stat().st_size

    elif algorithm == 'xz':
        with xz.open(str(path / 'compressed.xz'), 'wb') as f:
            f.write(encoded_data)
        compressed_size = (path / "compressed.xz").stat().st_size
    
    else:
        if temp_dir:
            temp_dir.cleanup()
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'gzip' or 'xz'.")

    if verbose:
        print(f"Original Size: {original_size}\nCompressed Size: {compressed_size}")

    if temp_dir:
        temp_dir.cleanup()

    return round(original_size / compressed_size, 3)