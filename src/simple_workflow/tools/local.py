"""Local (in-process) tools: ``unzip`` and ``audio``.

``audio`` implements three operations on 16-bit PCM WAV data using only the
standard library ``wave`` module and ``numpy``:

* ``mono``   - down-mix every input to a single channel,
* ``reverb`` - apply a lightweight FFT convolution reverb,
* ``mix``    - sum several streams together and normalise the peaks.
"""

from __future__ import annotations

import io
import os
import zipfile
from typing import Any

import numpy as np

from ..errors import ToolExecutionError
from ..media import to_wav
from ..signature import Param, Signature
from ..types import ValueType
from .base import RunContext, Tool

_TEXT = ValueType.TEXT
_BINARY = ValueType.BINARY


class UnzipTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            "unzip",
            Signature(
                params=(Param("path", _TEXT), Param("archive", _BINARY)),
                output=_BINARY,
            ),
        )

    async def run(self, params: list[Any], ctx: RunContext) -> bytes:  # noqa: ANN401
        path, archive = params[0], params[1]
        try:
            zf = zipfile.ZipFile(io.BytesIO(bytes(archive)))
        except zipfile.BadZipFile as exc:
            raise ToolExecutionError(f"unzip: not a valid zip archive: {exc}") from exc
        with zf:
            member = _find_member(zf.namelist(), path)
            if member is None:
                raise ToolExecutionError(
                    f"unzip: {path!r} not found in archive "
                    f"(members: {', '.join(zf.namelist()[:20])})"
                )
            return zf.read(member)


def _find_member(names: list[str], wanted: str) -> str | None:
    if wanted in names:
        return wanted
    lower = wanted.lower()
    for name in names:
        if name.lower() == lower:
            return name
    base = os.path.basename(lower)
    for name in names:
        if os.path.basename(name).lower() == base:
            return name
    # Fall back to matching on the stem, ignoring extensions, so that a request
    # for "noreverb" matches the archive member "noreverb.wav".
    want_stem = os.path.splitext(base)[0]
    for name in names:
        if os.path.splitext(os.path.basename(name).lower())[0] == want_stem:
            return name
    return None


class AudioTool(Tool):
    HOP = 0.5  # seconds of reverb tail

    def __init__(self) -> None:
        super().__init__(
            "audio",
            Signature(
                params=(Param("op", _TEXT),),
                variadic=Param("inputs", _BINARY),
                output=_BINARY,
            ),
        )

    async def run(self, params: list[Any], ctx: RunContext) -> bytes:  # noqa: ANN401
        op = str(params[0])
        streams = [bytes(p) for p in params[1:]]
        if not streams:
            raise ToolExecutionError(f"audio:{op} requires at least one input stream")
        try:
            if op == "mono":
                return self._mono(streams)
            if op == "reverb":
                return self._reverb(streams)
            if op == "mix":
                return self._mix(streams)
            if op in ("decode", "to-wav", "wav"):
                return self._decode(streams)
        except ToolExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolExecutionError(f"audio:{op} failed: {exc}") from exc
        raise ToolExecutionError(
            f"audio: unknown operation {op!r} (expected one of: mono, reverb, mix, decode)"
        )

    def _decode(self, streams: list[bytes]) -> bytes:
        return to_wav(streams[0])

    def _mono(self, streams: list[bytes]) -> bytes:
        samples, rate = _decode_wav(streams[0])
        mono = samples.mean(axis=1, keepdims=True)
        return _encode_wav(mono, rate)

    def _mix(self, streams: list[bytes]) -> bytes:
        decoded = [_decode_wav(s) for s in streams]
        rate = max(stream_rate for _, stream_rate in decoded)
        channels = max(samples.shape[1] for samples, _ in decoded)
        prepared: list[np.ndarray] = []
        for samples, stream_rate in decoded:
            aligned = _resample(samples, stream_rate, rate)
            if aligned.shape[1] < channels:
                if aligned.shape[1] == 1:
                    aligned = np.repeat(aligned, channels, axis=1)
                else:
                    aligned = np.pad(aligned, ((0, 0), (0, channels - aligned.shape[1])))
            prepared.append(aligned)

        length = max(samples.shape[0] for samples in prepared)
        mixed = np.zeros((length, channels), dtype=np.float32)
        for samples in prepared:
            mixed[: samples.shape[0]] += samples
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 1.0:
            mixed /= peak
        return _encode_wav(mixed, rate)

    def _reverb(self, streams: list[bytes]) -> bytes:
        samples, rate = _decode_wav(streams[0])
        impulse = _make_impulse(rate)
        wet = np.empty_like(samples)
        for channel in range(samples.shape[1]):
            wet[:, channel] = _fft_convolve(samples[:, channel], impulse)
        peak = float(np.max(np.abs(wet))) if wet.size else 0.0
        if peak > 1.0:
            wet /= peak
        output = 0.7 * samples + 0.5 * wet
        return _encode_wav(output, rate)


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------


def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(io.BytesIO(data), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())

    if width == 1:
        array = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        array = (array - 128.0) / 128.0
    elif width == 2:
        array = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        values = (values ^ 0x800000) - 0x800000
        array = values.astype(np.float32) / 8388608.0
    elif width == 4:
        array = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ToolExecutionError(f"unsupported WAV sample width: {width} bytes")

    if channels <= 0:
        raise ToolExecutionError("WAV file reports zero channels")
    return array.reshape(-1, channels), rate


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler used to align mixed streams."""
    if src_rate == dst_rate or samples.shape[0] == 0:
        return samples
    n_out = max(1, int(round(samples.shape[0] * dst_rate / src_rate)))
    src_x = np.linspace(0.0, 1.0, samples.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, 1.0, n_out, endpoint=False)
    out = np.empty((n_out, samples.shape[1]), dtype=np.float32)
    for channel in range(samples.shape[1]):
        out[:, channel] = np.interp(dst_x, src_x, samples[:, channel])
    return out


def _encode_wav(samples: np.ndarray, rate: int) -> bytes:
    import wave

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(pcm.shape[1])
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(pcm.tobytes())
        return buffer.getvalue()


def _make_impulse(rate: int) -> np.ndarray:
    length = max(1, int(AudioTool.HOP * rate))
    t = np.arange(length, dtype=np.float32)
    envelope: np.ndarray = np.exp(-6.0 * t / length)
    rng = np.random.default_rng(0)
    noise: np.ndarray = rng.standard_normal(length).astype(np.float32)
    impulse: np.ndarray = envelope * noise
    pre = min(length, max(1, int(0.01 * rate)))
    impulse[:pre] *= np.linspace(0.0, 1.0, pre, dtype=np.float32)
    peak = float(np.max(np.abs(impulse))) + 1e-9
    return impulse / peak


def _fft_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    size = len(signal) + len(impulse) - 1
    n = 1 << (size - 1).bit_length()
    spectrum = np.fft.rfft(signal, n) * np.fft.rfft(impulse, n)
    result: np.ndarray = np.asarray(np.fft.irfft(spectrum, n)[: len(signal)], dtype=np.float32)
    return result


def local_tools() -> list[Tool]:
    return [UnzipTool(), AudioTool()]
