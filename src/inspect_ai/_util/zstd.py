"""Zstandard compression from the standard library (PEP 784).

Python 3.14+ ships ``compression.zstd``; on earlier versions we use its
official backport, ``backports.zstd``, which has the same API. Import zstd
names from this module rather than from either package directly.
"""

import sys

if sys.version_info >= (3, 14):
    from compression.zstd import (
        CompressionParameter,
        ZstdCompressor,
        ZstdDecompressor,
        ZstdError,
        ZstdFile,
        compress,
        decompress,
    )
else:
    from backports.zstd import (
        CompressionParameter,
        ZstdCompressor,
        ZstdDecompressor,
        ZstdError,
        ZstdFile,
        compress,
        decompress,
    )


class ZstdMultiFrameDecompressor:
    """Decompresses a zstd stream that may hold several frames.

    ``ZstdDecompressor`` stops at the end of the first frame (``eof``) and
    refuses further input. This starts a new one on the bytes that follow, so
    a multi-frame stream (valid per RFC 8878) decodes as one. ``decompress``
    takes ``max_length`` and the object reports ``needs_input``, with the same
    meaning as for ``ZstdDecompressor``. Data that ends mid-frame is not an
    error here; callers that must detect truncation check it themselves.
    """

    def __init__(self) -> None:
        self._obj = ZstdDecompressor()
        # bytes after the end of a frame, not yet given to a decompressor
        self._input = b""

    @property
    def needs_input(self) -> bool:
        """Whether more output needs more input (``False``: call with ``b""``)."""
        return self._obj.needs_input and not self._input

    def decompress(self, data: bytes, max_length: int = -1) -> bytes:
        """Decompress ``data``, returning at most ``max_length`` bytes if it is >= 0."""
        if self._input:
            data = self._input + data
            self._input = b""
        pieces: list[bytes] = []
        size = 0
        while max_length < 0 or size < max_length:
            if not self._obj.needs_input:
                chunk = b""
            elif data:
                chunk, data = data, b""
            else:
                break
            piece = self._obj.decompress(
                chunk, -1 if max_length < 0 else max_length - size
            )
            pieces.append(piece)
            size += len(piece)
            if self._obj.eof:
                data = self._obj.unused_data + data
                self._obj = ZstdDecompressor()
        self._input = data
        return b"".join(pieces)


__all__ = [
    "CompressionParameter",
    "ZstdCompressor",
    "ZstdDecompressor",
    "ZstdError",
    "ZstdFile",
    "ZstdMultiFrameDecompressor",
    "compress",
    "decompress",
]
