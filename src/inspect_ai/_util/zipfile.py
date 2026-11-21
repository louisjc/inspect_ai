"""ZIP file helpers and monkey-patches for zstd support.

Python 3.14+ ``zipfile`` supports zstd (method 93) natively. On earlier
versions we register the method on the stdlib ``zipfile`` module ourselves,
backed by ``backports.zstd``.

Additionally, we install a second monkey-patch on ``zipfile._get_compressor``
that caps each emitted zstd frame at ``_MAX_INPUT_PER_FRAME`` bytes of input.
Large single zstd frames (>= 256 MiB compressed) trigger an overflow bug in
the pure-JS ``fzstd`` decoder used by our viewers; multi-framing keeps every
frame well under that threshold.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from typing import Any, BinaryIO

import anyio
import anyio.from_thread

from inspect_ai._util.zstd import (
    CompressionParameter,
    ZstdCompressor,
    ZstdMultiFrameDecompressor,
)

logger = logging.getLogger(__name__)


def _register_zip_zstandard() -> int:
    """Register zstd (method 93) on the stdlib ``zipfile`` module (Python < 3.14).

    Defines ``ZIP_ZSTANDARD`` and the method's name, and makes
    ``_check_compression`` accept it, as Python 3.14 ``zipfile`` does.
    Compressor and decompressor come from ``_install_multiframe_patches``.
    Member headers keep zipfile's default "version needed to extract", as
    they did with the ``zipfile-zstd`` package this replaces. Idempotent.
    """
    method = 93
    # zipfile's private names are not in its type stubs
    module = vars(zipfile)
    if module.get("ZIP_ZSTANDARD") == method:
        return method
    module["compressor_names"][method] = "zstd"
    original_check = module["_check_compression"]

    def check_compression(compression: int) -> None:
        if compression != method:
            original_check(compression)

    module["_check_compression"] = check_compression
    module["ZIP_ZSTANDARD"] = method
    return method


if sys.version_info >= (3, 14):
    _ZIP_ZSTANDARD: int = zipfile.ZIP_ZSTANDARD
else:
    _ZIP_ZSTANDARD = _register_zip_zstandard()

zipfile_compress_kwargs: dict[str, Any] = {
    "compression": _ZIP_ZSTANDARD,
    "compresslevel": None,
}


# Keep these zstd settings under the viewer's decoder limits.
#
# Shard merge (inspect_ai.log._shards) now copies sample members into the
# merged log as-is, without recompressing them. A frame written here is
# the frame the viewer decodes.
#
# The viewer worker cannot decode:
# - A history window over 32 MiB. In
#   src/inspect_ai/_view/ts-mono/apps/inspect/src/client/remote/zstd-decoder.ts,
#   MAX_WINDOW_LOG is 25 (2^25 = 32 MiB) at line 8, and scanFrames raises
#   ZstdWindowSizeError at lines 106-107 when windowSize > 2**25. For a
#   single-segment frame, the window is the frame's content size (line 105).
# - A single zstd frame whose compressed size is >= 256 MiB, which overflows
#   the pure-JS fzstd decoder the worker uses (see the module docstring
#   above). The viewer does not itself enforce a 200 MiB cap.
#   _MAX_INPUT_PER_FRAME (200 MiB of uncompressed input) is this writer's
#   cap, chosen so compressed frames stay under that overflow.
#
# Default compresslevel 3 (when compresslevel is None below) selects a
# window well under 32 MiB. Do not raise the level, set a larger windowLog,
# or raise _MAX_INPUT_PER_FRAME, without staying under those limits.
_MAX_INPUT_PER_FRAME = 200 * 1024 * 1024

# Worker threads for zstd compression, as zipfile_zstd used before; capped at
# what libzstd supports (0 when it is built without multithreading).
_ZSTD_THREADS = 12


def _zstd_compressor(level: int) -> ZstdCompressor:
    options: dict[int, int] = {CompressionParameter.compression_level: level}
    max_workers = CompressionParameter.nb_workers.bounds()[1]
    if max_workers > 0:
        options[CompressionParameter.nb_workers] = min(_ZSTD_THREADS, max_workers)
    return ZstdCompressor(options=options)


class _MultiFrameZstdCompressObj:
    """A zstd compressobj that chunks its output into multiple frames.

    Ends the current frame (``FLUSH_FRAME``) every ``_MAX_INPUT_PER_FRAME``
    bytes of input; the compressor then starts a new frame on the next input.
    Multi-frame zstd streams are valid per spec -- any compliant decoder reads
    them transparently.
    """

    def __init__(self, compressor: ZstdCompressor) -> None:
        self._compressor = compressor
        self._input_bytes = 0

    def compress(self, data: bytes) -> bytes:
        view = memoryview(data)
        pieces: list[bytes] = []
        offset = 0
        n = len(view)
        while offset < n:
            remaining_cap = _MAX_INPUT_PER_FRAME - self._input_bytes
            end = min(offset + remaining_cap, n)
            pieces.append(self._compressor.compress(view[offset:end]))
            self._input_bytes += end - offset
            offset = end
            if self._input_bytes >= _MAX_INPUT_PER_FRAME:
                pieces.append(self._compressor.flush(ZstdCompressor.FLUSH_FRAME))
                self._input_bytes = 0
        return b"".join(pieces)

    def flush(self) -> bytes:
        # If the last ``compress()`` call landed exactly on the frame boundary,
        # the frame is already ended. Flushing again would append an empty
        # trailing frame.
        if self._input_bytes == 0:
            return b""
        self._input_bytes = 0
        return self._compressor.flush(ZstdCompressor.FLUSH_FRAME)


class _MultiFrameZstdDecompressObj(ZstdMultiFrameDecompressor):
    """A zstd decompressobj for ``zipfile`` that spans multiple frames.

    The compressor splits large entries into multiple frames (see
    ``_MultiFrameZstdCompressObj``). The stdlib ``zipfile._read1`` path for
    non-deflate compression does:

        data = self._decompressor.decompress(data)
        self._eof = self._decompressor.eof or self._compress_left <= 0

    We return ``eof=False`` always so that ``self._eof`` is driven purely by
    ``self._compress_left <= 0`` (all compressed bytes fed).

    Since CPython gh-156002, ``_read1`` calls ``decompress(data, max_length)``
    and reads more compressed bytes only while ``needs_input`` is True (draining
    with ``decompress(b"")`` otherwise); both are provided by the base class.
    """

    def flush(self) -> bytes:
        return b""

    @property
    def eof(self) -> bool:
        # Always False: let compress_left drive the outer EOF check.
        return False


def _install_multiframe_patches() -> None:
    """Install multi-frame zstd compressor and decompressor patches.

    Idempotent. Wraps the stdlib ``_get_compressor`` / ``_get_decompressor``
    and handles zstd itself on every Python version.
    """
    if getattr(zipfile, "_inspect_ai_multiframe_installed", False):
        return

    original_compressor = zipfile._get_compressor  # type: ignore[attr-defined]
    original_decompressor = zipfile._get_decompressor  # type: ignore[attr-defined]

    def patched_compressor(compress_type: int, compresslevel: int | None = None) -> Any:
        if compress_type == _ZIP_ZSTANDARD:
            # Level and (implicit) window: stay under the viewer limits
            # documented at _MAX_INPUT_PER_FRAME above. The merge copies
            # these frames as-is, without recompressing them.
            level = 3 if compresslevel is None else compresslevel
            return _MultiFrameZstdCompressObj(_zstd_compressor(level))
        return original_compressor(compress_type, compresslevel)

    def patched_decompressor(compress_type: int) -> Any:
        if compress_type == _ZIP_ZSTANDARD:
            return _MultiFrameZstdDecompressObj()
        return original_decompressor(compress_type)

    zipfile._get_compressor = patched_compressor  # type: ignore[attr-defined]
    zipfile._get_decompressor = patched_decompressor  # type: ignore[attr-defined]
    zipfile._inspect_ai_multiframe_installed = True  # type: ignore[attr-defined]


_install_multiframe_patches()


__all__ = ["zipfile_compress_kwargs"]


# ---------------------------------------------------------------------------
# Archive rewriting helpers
#
# An append-mode ZipFile never removes bytes: a member "pruned" from the
# central directory, or superseded by a later write under the same name, stays
# in the member area as dead bytes. These helpers measure that and rewrite an
# archive down to its live members. They know nothing about eval logs; the
# recorders decide which members are live and when a rewrite is worth it.


def copy_live_members(
    src: zipfile.ZipFile,
    dst: zipfile.ZipFile,
    exclude: frozenset[str] = frozenset(),
    *,
    cancellable: bool = False,
) -> None:
    """Copy each name's last member from ``src`` to ``dst``, streaming.

    Dedupes by member name, last entry winning — a duplicate name is how a
    writer supersedes a member (read-by-name resolves to the last entry), and
    copying every info would write those superseded bytes twice. Opening the
    destination entry with the source ``ZipInfo`` preserves the original
    compression type / date_time / external_attr; the data still round-trips
    through decompress + recompress, streamed in chunks so a large member
    never sits in memory whole. Blocking — run in a worker thread when called
    from the event loop. ``cancellable`` requires an AnyIO worker and checks
    its host task's cancellation between chunks.
    """
    infos = {info.filename: info for info in src.infolist()}
    for info in infos.values():
        if info.filename in exclude:
            continue
        with (
            src.open(info, "r") as reader,
            dst.open(info, "w", force_zip64=True) as writer,
        ):
            while True:
                if cancellable:
                    anyio.from_thread.check_cancelled()
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                writer.write(chunk)


class RawMemberWriter:
    """Appends a member to a ``ZipFile`` from its compressed bytes.

    The bytes are written as given, with no decompression or recompression,
    so they must be the complete compressed data of one member in
    ``compress_type`` (for example a member of another archive, read with
    ``AsyncZipReader.open_member_raw``). :meth:`finish` records the CRC-32
    and size of the uncompressed data, which the caller supplies. The local
    header always has ZIP64 fields, as for ``ZipFile.open(..., force_zip64=True)``;
    the central directory uses ZIP64 where a size or offset needs it. Like
    ``ZipFile.open`` for writing, no other member can be written until the
    writer is closed. Blocking local I/O: run it in a worker thread from
    async code.
    """

    def __init__(self, zf: zipfile.ZipFile, name: str, compress_type: int) -> None:
        """Start the member ``name`` in ``zf``.

        Args:
            zf: An archive open for writing, to a seekable file.
            name: The member's name.
            compress_type: The compression method of the bytes to be written.

        Raises:
            ValueError: ``zf`` does not write to a seekable file, or cannot
                take a member now (closed, read-only or with a member open).
        """
        # zipfile wraps an unseekable file in an object with no `seekable`
        seekable = getattr(zf.fp, "seekable", None)
        if seekable is None or not seekable():
            raise ValueError(
                f"Cannot write member {name} from its compressed bytes: the "
                "archive must be open for writing to a seekable file."
            )
        self._zf = zf
        self._compress_type = compress_type
        # written as stored bytes, then given its method, CRC-32 and size
        self._info = zipfile.ZipInfo(name, date_time=time.localtime(time.time())[:6])
        self._info.compress_type = zipfile.ZIP_STORED
        self._writer = zf.open(self._info, "w", force_zip64=True)
        self._header_size = len(self._info.FileHeader(zip64=True))

    def write(self, data: bytes) -> None:
        """Append compressed bytes of the member."""
        self._writer.write(data)

    def finish(self, crc32: int, file_size: int) -> None:
        """Close the member, recording its uncompressed data's CRC-32 and size."""
        if self._writer.closed:
            raise ValueError(f"Member {self._info.filename} is already closed.")
        self._writer.close()
        info = self._info
        info.compress_type = self._compress_type
        info.CRC = crc32
        info.file_size = file_size
        header = info.FileHeader(zip64=True)
        assert len(header) == self._header_size, "the local header changed size"
        fp = self._zf.fp
        assert fp is not None
        fp.seek(info.header_offset)
        fp.write(header)
        fp.seek(self._zf.start_dir)

    @property
    def closed(self) -> bool:
        """Whether the member is closed, by :meth:`finish` or :meth:`close`."""
        return self._writer.closed

    def close(self) -> None:
        """Close the member if :meth:`finish` did not; does nothing otherwise.

        A member closed this way is recorded as stored bytes, so discard the
        archive. Use it to release the archive after an error.
        """
        self._writer.close()


def compact_zip(src_file: BinaryIO, live: frozenset[str]) -> BinaryIO:
    """Copy the live members of a closed zip temp file into a fresh temp file.

    Live means the last member under each name in ``live``; pruned and
    superseded members are left behind. Take ``live`` from the writer's
    in-memory central directory rather than the file's: ``ZipFile.close``
    rewrites the on-disk directory only after a write, so a directory-only
    prune since the last write is not on disk yet and the closed file's
    directory may still list pruned members. Blocking (decompress +
    recompress of every member) — run in a worker thread created by AnyIO.
    Cancellation is checked between chunks and closes the incomplete output
    before propagating to the caller.
    """
    src_file.seek(0)
    out: BinaryIO = tempfile.TemporaryFile()
    try:
        with (
            zipfile.ZipFile(src_file, "r") as src,
            zipfile.ZipFile(out, "w", **zipfile_compress_kwargs) as dst,
        ):
            copy_live_members(
                src, dst, exclude=frozenset(src.namelist()) - live, cancellable=True
            )
    except BaseException:
        out.close()
        raise
    return out


def zip_needs_rewrite(file: BinaryIO, is_live: Callable[[str], bool]) -> bool:
    """Whether an archive holds members ``is_live`` rejects, or bytes no member accounts for.

    Answers "can this file be adopted as-is, or must it be rewritten to shed
    what it should not carry" without inflating any member: a member outside
    the live set, or a gap between members (a pruned member's leftover
    bytes, which ordinary readers no longer list but still recoverable from
    the file), both require a rewrite. Walks exact local header lengths —
    estimating from central-directory extras can miss small gaps, especially
    with ZIP64 — and conservatively requires a rewrite for data descriptors
    or any layout it cannot verify. Blocking local I/O — run in a worker
    thread.
    """
    file.seek(0)
    with zipfile.ZipFile(file, "r") as archive:
        if any(not is_live(name) for name in archive.NameToInfo):
            return True
        end = 0
        for info in sorted(archive.NameToInfo.values(), key=lambda i: i.header_offset):
            if info.header_offset != end or info.flag_bits & 0x08:
                return True
            file.seek(info.header_offset)
            header = file.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                return True
            name_length = int.from_bytes(header[26:28], "little")
            extra_length = int.from_bytes(header[28:30], "little")
            end += 30 + name_length + extra_length + info.compress_size
        return end != archive.start_dir


def zip_dead_bytes(archive: zipfile.ZipFile) -> int:
    """Bytes of the member area no live member accounts for (0 for a fresh archive).

    The member area runs from offset 0 to ``start_dir`` (where the central
    directory is written at close); whatever the live members' local headers
    and compressed data don't cover is dead — pruned or superseded members,
    including any inherited when the archive was copied from another. A local
    header is reconstructed as ``FileHeader(zip64=True)``: exact for members
    written with ``force_zip64``, and 20 bytes over for ``writestr`` members,
    so the count is if anything an under-estimate.
    """
    live = sum(
        len(info.FileHeader(zip64=True)) + info.compress_size
        for info in archive.NameToInfo.values()
    )
    return max(0, archive.start_dir - live)
