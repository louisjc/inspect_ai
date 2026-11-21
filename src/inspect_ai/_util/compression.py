import zlib
from collections.abc import AsyncIterator, Callable, Generator
from typing import Protocol

from .zip_common import ZipCompressionMethod
from .zstd import ZstdError, ZstdMultiFrameDecompressor


class Decompressor(Protocol):
    """Protocol for async decompressors that read from a stream iterator."""

    @property
    def exhausted(self) -> bool:
        """Whether the decompressor has finished processing all input."""
        ...

    async def decompress_next(self, stream_iterator: AsyncIterator[bytes]) -> bytes:
        """Read compressed chunks until decompressed output is available.

        The decompressor may buffer multiple input chunks before producing output,
        as it needs to accumulate enough compressed data to decode a full block.
        This method keeps reading from the source stream until decompression
        yields non-empty output.

        Raises:
            StopAsyncIteration: When the stream is exhausted.
        """
        ...


class ZstdDecompressor(Decompressor):
    """Decompressor for zstd compressed data, supporting multi-frame streams."""

    def __init__(self) -> None:
        self._obj = ZstdMultiFrameDecompressor()
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    async def decompress_next(self, stream_iterator: AsyncIterator[bytes]) -> bytes:
        """Read compressed chunks until decompressed output is available."""
        if self._exhausted:
            raise StopAsyncIteration
        while True:
            try:
                data = await stream_iterator.__anext__()
            except StopAsyncIteration:
                # decompress() returns all the output its input allows, so
                # nothing is left to flush
                self._exhausted = True
                raise
            decompressed = self._obj.decompress(data)
            if decompressed:
                return decompressed


class DeflateDecompressor(Decompressor):
    """Decompressor for DEFLATE (raw) compressed data."""

    def __init__(self) -> None:
        self._decompressor: zlib._Decompress | None = None
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    async def decompress_next(self, stream_iterator: AsyncIterator[bytes]) -> bytes:
        """Read compressed chunks until decompressed output is available."""
        if self._decompressor is None:
            self._decompressor = zlib.decompressobj(-15)  # Raw DEFLATE
        while True:
            try:
                chunk = await stream_iterator.__anext__()
                decompressed = self._decompressor.decompress(chunk)
                if decompressed:
                    return decompressed
            except StopAsyncIteration:
                final = self._decompressor.flush()
                self._decompressor = None
                self._exhausted = True
                if final:
                    return final
                raise


class PieceDecompressor:
    """Decompress a ZIP member's data, pushed in chunks, into bounded pieces.

    Passes the decompressed bytes to `sink` in pieces of at most `piece_size`
    bytes, however far the data expands, so a caller that handles each piece
    as it arrives holds only one at a time. Multi-frame zstd data is read
    across its frames. Bytes after the end of a deflate stream are refused
    (`zlib.error`). Blocking (CPU-bound): run it in a worker
    thread from async code.
    """

    def __init__(
        self,
        method: ZipCompressionMethod,
        sink: Callable[[bytes], None],
        piece_size: int = 1024 * 1024,
    ) -> None:
        self._method = method
        self._sink = sink
        self._piece_size = piece_size
        self._deflate: zlib._Decompress | None = None
        self._zstd: ZstdMultiFrameDecompressor | None = None
        self._zstd_frames: _ZstdFrames | None = None
        if method == ZipCompressionMethod.DEFLATE:
            self._deflate = zlib.decompressobj(-15)  # Raw DEFLATE
        elif method == ZipCompressionMethod.ZSTD:
            self._zstd = ZstdMultiFrameDecompressor()
            # the decompressor does not report data that ends mid-frame
            self._zstd_frames = _ZstdFrames()
        elif method != ZipCompressionMethod.STORED:
            raise NotImplementedError(f"Unsupported compression method: {method}")

    def write(self, data: bytes) -> None:
        """Decompress the next chunk of the member's data.

        Raises:
            zlib.error: Bytes follow the end of a deflate stream.
        """
        if self._zstd is not None:
            chunk = data
            # a full piece may leave more output pending
            while piece := self._zstd.decompress(chunk, self._piece_size):
                self._sink(piece)
                chunk = b""
                if self._zstd.needs_input:
                    break
            assert self._zstd_frames is not None
            self._zstd_frames.write(data)
        elif self._deflate is not None:
            # Bytes after the stream end used to be ignored; the merge now
            # publishes the compressed bytes as they are, so refuse them.
            if self._deflate.eof and data:
                raise zlib.error(
                    "Error -5 while decompressing data: bytes after the end of a deflate stream"
                )
            while not self._deflate.eof:
                piece = self._deflate.decompress(data, self._piece_size)
                if piece:
                    self._sink(piece)
                data = self._deflate.unconsumed_tail
                # a full piece may leave more output pending
                if not data and len(piece) < self._piece_size:
                    break
            if self._deflate.eof and self._deflate.unused_data:
                raise zlib.error(
                    "Error -5 while decompressing data: bytes after the end of a deflate stream"
                )
        else:
            for start in range(0, len(data), self._piece_size):
                self._sink(data[start : start + self._piece_size])

    def finish(self) -> None:
        """Check that the data held a complete stream.

        Raises:
            zlib.error: A deflate stream ended before its final block, as
                `zlib.decompress` raises.
            ZstdError: A zstd frame ended before its last block or its
                checksum (`compression.zstd.ZstdError`; `backports.zstd.ZstdError`
                before Python 3.14).
        """
        if self._deflate is not None and not self._deflate.eof:
            raise zlib.error(
                "Error -5 while decompressing data: incomplete or truncated stream"
            )
        if self._zstd_frames is not None:
            self._zstd_frames.finish()


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DICTIONARY_ID_SIZES = (0, 1, 2, 4)
_ZSTD_CONTENT_SIZE_SIZES = (0, 2, 4, 8)
_ZSTD_RLE_BLOCK = 1
_ZSTD_RESERVED_BLOCK = 3


class _ZstdFrames:
    """Follows the frame structure of zstd data, without decoding it.

    Reads each frame's header, block headers and checksum (RFC 8878), so that
    `finish` can tell whether the data ended between frames. Skippable frames
    are refused: Inspect never writes them.
    """

    def __init__(self) -> None:
        self._between_frames = True
        self._field = bytearray()
        self._fields = self._read_frames()
        # bytes of the next field to read, or (negative) to pass over
        self._want = next(self._fields)

    def write(self, data: bytes) -> None:
        position = 0
        while position < len(data):
            if self._want < 0:
                taken = min(-self._want, len(data) - position)
                self._want += taken
                position += taken
                if self._want == 0:
                    self._want = self._fields.send(b"")
            else:
                taken = min(self._want - len(self._field), len(data) - position)
                self._field += data[position : position + taken]
                position += taken
                if len(self._field) == self._want:
                    field = bytes(self._field)
                    self._field.clear()
                    self._want = self._fields.send(field)

    def finish(self) -> None:
        if not self._between_frames or self._field:
            raise ZstdError("incomplete or truncated zstd frame")

    def _read_frames(self) -> Generator[int, bytes, None]:
        while True:
            self._between_frames = True
            magic = yield 4
            self._between_frames = False
            if magic != _ZSTD_MAGIC:
                raise ZstdError("unsupported zstd frame")
            descriptor = (yield 1)[0]
            single_segment = bool(descriptor & 0x20)
            content_size_flag = descriptor >> 6
            header = (
                (0 if single_segment else 1)  # window descriptor
                + _ZSTD_DICTIONARY_ID_SIZES[descriptor & 0x03]
                + _ZSTD_CONTENT_SIZE_SIZES[content_size_flag]
                + (1 if single_segment and content_size_flag == 0 else 0)
            )
            if header:
                yield -header
            last = False
            while not last:
                block = int.from_bytes((yield 3), "little")
                last = bool(block & 0x01)
                block_type = (block >> 1) & 0x03
                if block_type == _ZSTD_RESERVED_BLOCK:
                    raise ZstdError("reserved zstd block type")
                content = 1 if block_type == _ZSTD_RLE_BLOCK else block >> 3
                if content:
                    yield -content
            if descriptor & 0x04:  # content checksum
                yield -4


def decompress_bytes(data: bytes, method: ZipCompressionMethod) -> bytes:
    """Decompress a complete buffer using the given ZIP compression method."""
    if method == ZipCompressionMethod.STORED:
        return data
    elif method == ZipCompressionMethod.DEFLATE:
        return zlib.decompress(data, -15)
    elif method == ZipCompressionMethod.ZSTD:
        return ZstdMultiFrameDecompressor().decompress(data)
    else:
        raise NotImplementedError(f"Unsupported compression method: {method}")


class Compressor(Protocol):
    """Protocol for async compressors that read from a stream iterator."""

    @property
    def exhausted(self) -> bool:
        """Whether the compressor has finished processing all input."""
        ...

    async def compress_next(self, stream_iterator: AsyncIterator[bytes]) -> bytes:
        """Read uncompressed chunks until compressed output is available.

        The compressor may buffer multiple input chunks before producing output,
        as it needs to accumulate enough data to form a compressed block.
        This method keeps reading from the source stream until compression
        yields non-empty output.

        Raises:
            StopAsyncIteration: When the stream is exhausted.
        """
        ...


class DeflateCompressor(Compressor):
    """Compressor for DEFLATE (raw) compressed data."""

    def __init__(self) -> None:
        # wbits=-15 produces raw DEFLATE (no zlib/gzip wrapper)
        self._compressor: zlib._Compress | None = zlib.compressobj(
            level=6,
            wbits=-15,
        )
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    async def compress_next(self, stream_iterator: AsyncIterator[bytes]) -> bytes:
        """Read uncompressed chunks until compressed output is available."""
        while True:
            try:
                chunk = await stream_iterator.__anext__()
                if self._compressor is None:
                    raise StopAsyncIteration
                compressed = self._compressor.compress(chunk)
                if compressed:
                    return compressed
                # If no compressed data yet (buffered), continue reading
            except StopAsyncIteration:
                # Input stream exhausted, flush any remaining data
                if self._compressor:
                    final = self._compressor.flush()
                    self._compressor = None
                    self._exhausted = True
                    if final:
                        return final
                raise
